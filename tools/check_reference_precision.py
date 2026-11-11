#!/usr/bin/env python3
"""Independently check reference CTest output with Decimal arithmetic."""

import argparse
import json
import re
import struct
import sys
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
HEADER = re.compile(r"Reference precision=(float|double) epsilon=(" + NUMBER +
                    r") long_double_digits=(\d+)")
EULER = re.compile(r"Reference Euler (\d+) steps, x=(" + NUMBER + r"), q=(" + NUMBER +
                   r"),.*?max_conservation_error=(" + NUMBER + r")")
NON_DYADIC = re.compile(r"Reference non-dyadic dt=(" + NUMBER + r"), x=(" + NUMBER +
                        r"), q=(" + NUMBER + r")")


def binary_decimal(text):
    # The printed 17 digits round-trip both binary32 (promoted exactly to
    # binary64) and binary64. Preserve that actual binary value in Decimal.
    return Decimal.from_float(float(text))


def float32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def run(log, precision):
    headers = HEADER.findall(log)
    if len(headers) != 1 or headers[0][0] != precision:
        raise ValueError("expected exactly one matching precision header")
    levels = EULER.findall(log)
    if len(levels) != 3 or {int(row[0]) for row in levels} != {16, 32, 64}:
        raise ValueError("expected exactly one Euler result for each of 16, 32, 64 steps")
    non_dyadic = NON_DYADIC.findall(log)
    if len(non_dyadic) != 1 or float(non_dyadic[0][0]) != 0.1:
        raise ValueError("expected exactly one non-dyadic dt=0.1 result")

    with localcontext() as ctx:
        ctx.prec = 80
        epsilon = Decimal(2) ** (-23 if precision == "float" else -52)
        if binary_decimal(headers[0][1]) != epsilon:
            raise ValueError("logged epsilon does not match requested precision")
        results = []
        analytic_errors = []
        for steps_text, x_text, q_text, max_cons_text in sorted(levels, key=lambda r: int(r[0])):
            steps = int(steps_text)
            dt = Decimal(1) / steps
            result = check_case(precision, steps, dt, x_text, q_text,
                                binary_decimal(max_cons_text), epsilon)
            results.append(result)
            analytic_errors.append(Decimal(result["analytic_error"]))
        orders = [(analytic_errors[i - 1] / analytic_errors[i]).ln() / Decimal(2).ln()
                  for i in range(1, len(analytic_errors))]
        if any(not Decimal("0.9") <= order <= Decimal("1.1") for order in orders):
            raise ValueError("observed Euler order outside [0.9, 1.1]")
        dt_text, x_text, q_text = non_dyadic[0]
        # Requested step is a C++ double, not exact decimal one tenth.
        dt = binary_decimal(dt_text)
        nonbinary = check_case(precision, 10, dt, x_text, q_text, None, epsilon)
        return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "oracle": "Python decimal, 80 significant digits",
                "precision": precision, "cpp_long_double_digits": int(headers[0][2]),
                "epsilon": str(epsilon), "levels": results,
                "observed_orders": [str(order) for order in orders],
                "non_dyadic": nonbinary}


def check_case(precision, steps, dt, x_text, q_text, max_conservation, epsilon):
    x, q = binary_decimal(x_text), binary_decimal(q_text)
    if precision == "float" and any(float32(float(value)) != float(value) for value in (x, q)):
        raise ValueError("logged state is not representable as float32")
    euler_x = (Decimal(1) - dt) ** steps
    euler_q = Decimal(1) - euler_x
    exact = (-dt * steps).exp()
    if precision == "double":
        x_bound = Decimal("1e-12") + Decimal("1e-10") * abs(euler_x)
        q_bound = Decimal("1e-12") + Decimal("1e-10") * abs(euler_q)
        conservation_bound = Decimal("1e-12")
    else:
        converted_dt = Decimal.from_float(float32(float(dt)))
        bound = 8 * steps * epsilon + steps * abs(converted_dt - dt)
        x_bound = q_bound = conservation_bound = bound
    x_error, q_error = abs(x - euler_x), abs(q - euler_q)
    conservation = abs(x + q - 1)
    if x_error > x_bound or q_error > q_bound or conservation > conservation_bound:
        raise ValueError(f"{steps} steps exceed independent error or conservation bound")
    if max_conservation is not None and max_conservation > conservation_bound:
        raise ValueError(f"{steps} steps exceed logged maximum conservation bound")
    return {"steps": steps, "requested_dt": str(dt), "x": str(x), "q": str(q),
            "euler_x": str(euler_x), "euler_q": str(euler_q), "analytic_x": str(exact),
            "x_euler_error": str(x_error), "q_euler_error": str(q_error),
            "analytic_error": str(abs(x - exact)), "x_bound": str(x_bound),
            "q_bound": str(q_bound), "conservation_error": str(conservation),
            "max_logged_conservation_error": (str(max_conservation)
                                               if max_conservation is not None else None),
            "conservation_bound": str(conservation_bound)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--precision", choices=("float", "double"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.log.read_text(), args.precision)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    except (OSError, ValueError, ArithmeticError, struct.error) as error:
        print(f"reference precision validation failed: {error}", file=sys.stderr)
        return 1
    print(f"reference precision validation passed: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
