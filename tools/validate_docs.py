#!/usr/bin/env python3
"""Validate the opt-in Nexora documentation against provisioned build tools."""

import argparse
import platform
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from validate_serial import COMMON_OPTIONS, PACKAGE_DIR, ROOT, executable, now, verify_network, write_json


INHERITANCE = {
    "nexora::ParticipantIF": (
        "nexora::reference::Participant", "nexora::sph::WcsphParticipant3D",
    ),
    "nexora::sph::SphBoundary::SolidWallBoundary3D": (
        "nexora::sph::SphBoundary::FixedWall3D",
        "nexora::sph::SphBoundary::PrescribedMovingWall3D",
    ),
}
PRIVATE_ATTRIBUTES = {
    "budget_": "detail::ResourceBudget",
    "step_": "std::optional<WcsphStep3D>",
    "accepted_": "std::unique_ptr<WcsphState3D>",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args, log, *, diagnostic=None):
    args = [str(arg) for arg in args]
    header = f"[{now()}] {shlex.join(args)}"
    print(header, flush=True)
    with log.open("a", encoding="utf-8") as output:
        print(header, file=output, flush=True)
        result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        output.write(result.stdout)
        print(f"[{now()}] exit {result.returncode}", file=output)
    if diagnostic:
        match = re.search(diagnostic, result.stdout, re.IGNORECASE | re.MULTILINE)
        require(result.returncode != 0 and match, f"expected a failed documentation diagnostic; see {log}")
        return {"exit": result.returncode, "diagnostic": match.group(0)}
    require(result.returncode == 0, f"command exited {result.returncode}; see {log}")
    return result.stdout.strip()


def inline_text(element):
    return " ".join("".join(element.itertext()).split())


def require_phrases(body, phrases, context):
    for phrase in phrases:
        require(phrase in body, f"{context}: missing documentation phrase {phrase!r}")


def description(element, phrases, context):
    body = " ".join(inline_text(part) for part in element
                    if part.tag in {"briefdescription", "detaileddescription"})
    require_phrases(body, phrases, context)
    return body.strip()


SVG = "{http://www.w3.org/2000/svg}"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
UML_LEGEND = "maintainer_uml_legend"
UML_LEGEND_LINK = f"index.html#{UML_LEGEND}"
DIAGRAM_ANCHORS = {"inheritance-overview", "inheritance-uml", "collaboration-overview", "collaboration-uml"}
UML_ACCESS_MARKER = re.compile(r"^[+\-#~*](?:\s|$)")


def read_svg(page, source):
    path = page.path.parent / unquote(urlsplit(source).path)
    root = ET.parse(path).getroot()
    require(root.tag == f"{SVG}svg" and root.get("viewBox"), f"{path}: invalid SVG diagram")
    return root


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.links = set()
        self.anchors = set()
        self.text_parts = []
        self.images = {}
        self.embedded = set()
        self.diagram_order = []
        self.attribute_rows = {}
        self.current_section = None
        self.current_row = None
        self.feed(path.read_text(encoding="utf-8"))
        self.close()

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        anchor = attributes.get("id") or attributes.get("name")
        if anchor:
            self.anchors.add(anchor)
            if anchor in DIAGRAM_ANCHORS:
                self.diagram_order.append(anchor)
        if tag == "a" and attributes.get("href"):
            self.links.add(attributes["href"])
        if tag == "a" and attributes.get("id") and attributes.get("name") == attributes["id"]:
            self.current_section = attributes["id"]
        if tag == "tr":
            self.current_row = next((item.removeprefix("memitem:") for item in attributes.get("class", "").split()
                                     if item.startswith("memitem:")), None)
            if self.current_row and self.current_section == "pri-attribs":
                self.attribute_rows[self.current_row] = []
        if tag == "img" and attributes.get("src"):
            self.images[attributes["src"]] = attributes.get("alt")
        asset = {"script": "src", "link": "href", "img": "src",
                 "iframe": "src", "embed": "src", "object": "data"}.get(tag)
        if asset and attributes.get(asset):
            url = urlsplit(attributes[asset])
            require(not url.scheme and not url.netloc and not url.path.startswith("/"),
                    f"{self.path}: expected a relative offline asset, found {attributes[asset]!r}")
            require((self.path.parent / unquote(url.path)).is_file(),
                    f"{self.path}: missing offline asset {attributes[asset]!r}")
            if tag in {"iframe", "embed", "object"}:
                self.embedded.add(attributes[asset])
                if re.search(r"_+(?:coll|inherit)_+(?:overview|graph)\.svg$", url.path):
                    self.diagram_order.append(attributes[asset])

    def handle_data(self, data):
        self.text_parts.append(data)
        if self.current_row in self.attribute_rows:
            self.attribute_rows[self.current_row].append(data)

    def handle_endtag(self, tag):
        if tag == "tr":
            self.current_row = None

    @property
    def text(self):
        return " ".join(" ".join(self.text_parts).split())


def graph_source(page, kind, suffix):
    matches = [source for source in page.embedded
               if re.search(rf"_+{kind}_+{suffix}\.svg$", urlsplit(source).path)]
    require(len(matches) == 1, f"{page.path}: expected one embedded {kind} {suffix}, found {matches}")
    return matches[0]


def svg_text(group):
    return {inline_text(element) for element in group.iter(f"{SVG}text")}


def svg_field_rows(node):
    rows = {}
    for element in node.iter(f"{SVG}text"):
        rows.setdefault(element.get("y"), []).append(inline_text(element))
    return {" ".join(parts) for parts in rows.values()}


def svg_node_label(node):
    parts = []
    for element in node.iter(f"{SVG}text"):
        part = inline_text(element)
        if UML_ACCESS_MARKER.match(part):
            break
        parts.append(part)
    return "".join(parts)


def check_svg_graph(page, source, origin, relations, *, overview=False, fields=None, edge_labels=None):
    svg = read_svg(page, source)
    groups = list(svg.iter(f"{SVG}g"))
    graph = next((group for group in groups if "graph" in group.get("class", "").split()), None)
    require(graph is not None and graph.findtext(f"{SVG}title") == origin,
            f"{source}: unexpected graph title for {origin}")
    rendered_nodes = [group for group in groups if "node" in group.get("class", "").split()]
    if overview:
        require(all(not any(UML_ACCESS_MARKER.match(text) for text in svg_text(node)) for node in rendered_nodes),
                f"{source}: overview shows UML member rows")
    nodes = {}
    for name, target in [(origin, None), *relations.items()]:
        label = name.rsplit("::", 1)[-1]
        if target:
            node = next((group for group in rendered_nodes
                         if any((link.get("href") or link.get(XLINK_HREF)) in {target, f"./{target}"}
                                for link in group.iter(f"{SVG}a"))), None)
            require(node is not None and (page.path.parent / target).is_file(),
                    f"{source}: {name} has no usable link to {target}")
        else:
            node = next((group for group in rendered_nodes if svg_node_label(group).rsplit("::", 1)[-1] == label),
                        None)
        require(node is not None and svg_node_label(node).rsplit("::", 1)[-1] == label,
                f"{source}: missing rendered node label {name}")
        nodes[name] = node
    origin_title = nodes[origin].findtext(f"{SVG}title")
    for name in relations:
        target_title = nodes[name].findtext(f"{SVG}title")
        require(origin_title and target_title, f"{source}: missing graph node identifier")
        directions = {f"{origin_title}->{target_title}", f"{target_title}->{origin_title}"}
        edge = next((group for group in groups if "edge" in group.get("class", "").split()
                     and group.findtext(f"{SVG}title") in directions
                     and any(path.get("d") for path in group.iter(f"{SVG}path"))
                     and any(polygon.get("points") for polygon in group.iter(f"{SVG}polygon"))), None)
        require(edge is not None, f"{source}: missing rendered edge connecting {origin} and {name}")
        if edge_labels and name in edge_labels:
            require(edge_labels[name] in {label.lstrip("+-#~ ") for label in svg_text(edge)},
                    f"{source}: missing rendered relation label {edge_labels[name]}")
    for name, (protection, member, datatype) in (fields or {}).items():
        # Doxygen may place the symbol and declaration in separate SVG text cells.
        expected = {f"{protection} {member}: {datatype}", f"{protection} {datatype} {member}"}
        require(expected & svg_field_rows(nodes[name]),
                f"{source}: {name} is missing UML field {protection} {member}: {datatype}")
    return (
        {node.findtext(f"{SVG}title"): svg_node_label(node) for node in rendered_nodes},
        {(link.get("href") or link.get(XLINK_HREF)) for node in rendered_nodes for link in node.iter(f"{SVG}a")
         if link.get("href") or link.get(XLINK_HREF)},
        {(group.findtext(f"{SVG}title"), frozenset(svg_text(group))) for group in groups
         if "edge" in group.get("class", "").split()},
    )


def check_graph_pair(page, kind, origin, relations, *, fields=None, edge_labels=None):
    overview = graph_source(page, kind, "overview")
    uml = graph_source(page, kind, "graph")
    section = {"coll": "collaboration", "inherit": "inheritance"}[kind]
    order = [f"{section}-overview", overview, f"{section}-uml", uml]
    require(any(page.diagram_order[offset:offset + len(order)] == order for offset in range(len(page.diagram_order))),
            f"{page.path}: expected the {section} overview immediately before its UML graph")
    require(UML_LEGEND_LINK in page.links, f"{page.path}: missing link to the maintainer UML legend")
    overview_relations = check_svg_graph(page, overview, origin, relations, overview=True, edge_labels=edge_labels)
    uml_relations = check_svg_graph(page, uml, origin, relations, fields=fields, edge_labels=edge_labels)
    require(overview_relations == uml_relations, f"{page.path}: {section} diagrams differ in nodes, edges or type links")
    return {"overview": overview, "uml": uml}


def check_docs(build, precision):
    docs = build / "docs"
    require(not any((docs / "html").glob("*__overview.svg")), f"{build}: internal graph assets in the public manual")
    index = ET.parse(docs / "xml/index.xml").getroot()
    compounds = {element.findtext("name"): element for element in index.findall("compound")
                 if element.get("kind") != "dir"}
    for header in ("Config.hpp", "Version.hpp"):
        require(header in compounds and compounds[header].get("kind") == "file",
                f"{build}: missing generated header in the API index: {header}")
    for name in compounds:
        require(not name.startswith(("nexora::sph::", "nexora::reference::"))
                and name not in {"nexora::sph", "nexora::reference"},
                f"{build}: internal implementation in the public API index: {name}")
    for name in index.findall(".//name"):
        require(not re.search(r"(?:^|::)detail(?:::|$)|\bFieldIssuer\b", name.text or ""),
                f"{build}: internal symbol in the API index: {name.text}")

    def compound(name):
        entry = compounds.get(name)
        require(entry is not None, f"{build}: missing API compound {name}")
        refid = entry.get("refid")
        return entry, ET.parse(docs / "xml" / f"{refid}.xml").find("compounddef")

    def member(owner, name, kind, protection=None):
        entry, definition = owner
        matches = [item for item in definition.findall(".//memberdef")
                   if item.findtext("name") == name and item.get("kind") == kind]
        refs = {item.get("refid") for item in entry.findall("member") if item.findtext("name") == name}
        require(matches and all(item.get("id") in refs for item in matches),
                f"{build}: missing indexed member {entry.findtext('name')}::{name}")
        require(protection is None or all(item.get("prot") == protection for item in matches),
                f"{build}: expected {entry.findtext('name')}::{name} to be {protection}")
        return matches

    namespace = compound("nexora")
    member(namespace, "FieldId", "typedef")
    real, = member(namespace, "real", "typedef")
    actual = real.findtext("type")
    require(actual == precision, f"{build}: expected nexora::real = {precision}, found {actual}")
    alias_page_id, anchor = real.get("id").rsplit("_1", 1)
    alias_page = Page(docs / "html" / f"{alias_page_id}.html")
    aliases = re.findall(r"\busing\s+(?:nexora::\s*)?real\s*=\s*(?:typedef\s+)?(\w+)", alias_page.text)
    require(anchor in alias_page.anchors and aliases and set(aliases) == {precision},
            f"{alias_page.path}: expected a linked nexora::real declaration for {precision}; found {aliases}")

    home = Page(docs / "html/index.html")
    require(f"{alias_page.path.name}#{anchor}" in home.links,
            f"{home.path}: missing link to nexora::real")
    public_pages = {}
    for name in ("Simulation", "ParticipantIF", "ReadField"):
        entry, _ = compound(f"nexora::{name}")
        page = Page(docs / "html" / f"{entry.get('refid')}.html")
        require(page.path.name in home.links, f"{home.path}: missing link to {name}")
        public_pages[name] = page

    simulation = compound("nexora::Simulation")
    for page in public_pages.values():
        require("Collaboration graph" not in page.images.values() and not page.diagram_order and not page.embedded,
                f"{page.path}: internal relationship diagram in the public reference")
    lifecycle = description(simulation[1], ("initialize", "run", "finalize"), "Simulation lifecycle")
    schedule = next((item for item in member(simulation, "run", "function", "public")
                     if item.findtext("param/declname") == "steps"), None)
    require(schedule is not None, f"{build}: missing Simulation::run schedule overload")
    run_body = description(schedule, ("last accepted participant state and clock", "finalize"), "Simulation::run")

    participant = compound("nexora::ParticipantIF")
    step, = member(participant, "step", "function", "public")
    do_step, = member(participant, "do_step", "function", "protected")
    step_body = description(step, ("finite and positive", "current accepted time constraint", "rounded endpoint"),
                            "ParticipantIF::step")
    hook_body = description(do_step, ("complete requested step", "next state's time constraint",
                                     "release temporary state", "preserve the previous accepted results"),
                            "ParticipantIF::do_step")
    for method in (step, do_step):
        page = public_pages["ParticipantIF"]
        fragment = method.get("id").rsplit("_1", 1)[1]
        require(fragment in page.anchors and f"{page.path.name}#{fragment}" in home.links,
                f"{home.path}: missing link to {method.findtext('name')}")

    read_field = public_pages["ReadField"]
    read_field_contract = (
        "Copying the Kokkos View does not extend access authorization.",
        "Keep the binding alive with an active lease until every operation using the View completes.",
    )
    read_field_excerpt = " ".join(read_field_contract)
    require_phrases(read_field.text, (read_field_excerpt,), read_field.path)
    for name, phrases in {
        "ReadLease": ("Finish all reads", "before releasing the lease"),
        "WriteLease": ("Finish all work", "before releasing the lease"),
        "WriteField": ("active lease", "until all work that uses the View completes"),
    }.items():
        _, definition = compound(f"nexora::{name}")
        description(definition, phrases, name)

    return {
        "index": str(home.path),
        "api_pages": {name: str(page.path) for name, page in public_pages.items()},
        "real_declaration": f"{alias_page.path}#{anchor}",
        "real_type": actual,
        "contracts": {"Simulation": lifecycle, "Simulation::run": run_body,
                      "ParticipantIF::step": step_body, "ParticipantIF::do_step": hook_body,
                      "ReadField HTML excerpt": read_field_excerpt},
    }


def check_collaboration(page, definition, compounds, origin, usage, *, bases=(), fields=None):
    origin_id = compounds[origin]
    expected = {member: compounds[name] for member, name in usage.items()}
    members = {item.findtext("name"): item.find("type/ref").get("refid")
               for item in definition.findall("sectiondef/memberdef")
               if item.get("kind") == "variable" and item.find("type/ref") is not None}
    require(expected.items() <= members.items(), f"{page.path}: missing collaboration members in XML")
    nodes = {item.find("link").get("refid"): item for item in definition.findall("collaborationgraph/node")
             if item.find("link") is not None}
    root = nodes.get(origin_id)
    require(root is not None, f"{page.path}: missing collaboration graph in XML")
    refs = {item.get("id"): refid for refid, item in nodes.items()}
    edges = {edge.findtext("edgelabel", "").lstrip("+-#~ "): refs.get(edge.get("refid"))
             for edge in root.findall("childnode") if edge.get("relation") == "usage"}
    require(expected.items() <= edges.items(), f"{page.path}: unexpected collaboration relations {edges}")
    inherited = {refs.get(edge.get("refid")) for edge in root.findall("childnode")
                 if edge.get("relation") == "public-inheritance"}
    require({compounds[name] for name in bases} <= inherited, f"{page.path}: missing base in collaboration XML")
    targets = {name: f"{compounds[name]}.html" for name in (*usage.values(), *bases)}
    return check_graph_pair(page, "coll", origin, targets, fields=fields,
                            edge_labels={name: member for member, name in usage.items()})


def check_internal_docs(build):
    docs = build / "docs-internal"
    index = ET.parse(docs / "xml/index.xml").getroot()
    compounds = {entry.findtext("name"): entry.get("refid") for entry in index.findall("compound")
                 if entry.get("kind") in {"class", "struct"}}
    require("nexora::detail::FieldIssuer" in compounds, f"{build}: missing the internal field issuer")
    home = Page(docs / "html/index.html")
    require(UML_LEGEND in home.anchors, f"{home.path}: missing anchor for the maintainer UML legend")
    require_phrases(home.text, ("Read a UML diagram", "+ marks public members", "# marks protected members",
                               "- marks private members", "open triangle points to the base type",
                               "open diamond marks the class that declares the member"), home.path)
    definitions = {}
    pages = {}
    for base, derived in INHERITANCE.items():
        for name in (base, *derived):
            refid = compounds.get(name)
            require(refid, f"{build}: missing internal class {name}")
            definitions[name] = ET.parse(docs / "xml" / f"{refid}.xml").find("compounddef")
            pages[name] = Page(docs / "html" / f"{refid}.html")

    inheritance_images = {}
    wcsph = "nexora::sph::WcsphParticipant3D"
    wcsph_fields = {wcsph: ("-", "max_cells_", "std::size_t")}
    for base, derived in INHERITANCE.items():
        base_page = pages[base]
        derived_refs = {entry.get("refid") for entry in definitions[base].findall("derivedcompoundref")
                        if entry.get("prot") == "public"}
        require({compounds[name] for name in derived} <= derived_refs,
                f"{build}: missing direct public derivations of {base}: {derived_refs}")
        require(base_page.path.name in home.links, f"{home.path}: missing link to {base}")
        for name in derived:
            derived_page = pages[name]
            direct_bases = {entry.get("refid") for entry in definitions[name].findall("basecompoundref")
                            if entry.get("prot") == "public"}
            require(compounds[base] in direct_bases, f"{build}: missing direct public base {base} for {name}")
            inheritance_images[name] = check_graph_pair(derived_page, "inherit", name, {base: base_page.path.name},
                                                       fields=wcsph_fields if name == wcsph else None)
        inheritance_images[base] = check_graph_pair(base_page, "inherit", base,
                                                   {name: pages[name].path.name for name in derived})

    wcsph_page = pages[wcsph]
    require(wcsph_page.path.name in home.links, f"{home.path}: missing link to {wcsph}")
    require("#pri-attribs" in wcsph_page.links and "Private Attributes" in wcsph_page.text,
            f"{wcsph_page.path}: missing Private Attributes section link")
    private_fields = definitions[wcsph].findall("sectiondef[@kind='private-attrib']/memberdef[@kind='variable']")
    xml_anchors = {field.get("id").rsplit("_1", 1)[1] for field in private_fields}
    require(wcsph_page.attribute_rows.keys() == xml_anchors,
            f"{wcsph_page.path}: Private Attributes table differs from the declared private fields")
    member_types = {}
    for name, expected_type in PRIVATE_ATTRIBUTES.items():
        member = next((item for item in definitions[wcsph].findall("sectiondef/memberdef")
                       if item.findtext("name") == name and item.get("kind") == "variable"), None)
        require(member is not None and member.get("prot") == "private",
                f"{build}: missing private attribute {wcsph}::{name}")
        actual_type = inline_text(member.find("type")).replace(" ", "")
        require(actual_type == expected_type, f"{build}: {wcsph}::{name} has type {actual_type}")
        anchor = member.get("id").rsplit("_1", 1)[1]
        row = "".join("".join(wcsph_page.attribute_rows.get(anchor, ())).split())
        require(anchor in wcsph_page.anchors and row.partition("=")[0] == f"{expected_type}{name}",
                f"{wcsph_page.path}: {name} and its type are missing from Private Attributes")
        member_types[name] = actual_type

    simulation = "nexora::Simulation"
    simulation_id = compounds[simulation]
    simulation_page = Page(docs / "html" / f"{simulation_id}.html")
    simulation_definition = ET.parse(docs / "xml" / f"{simulation_id}.xml").find("compounddef")
    participant = "nexora::ParticipantIF"
    time_state = "nexora::TimeState"
    collaboration = {
        simulation: check_collaboration(simulation_page, simulation_definition, compounds, simulation,
                                        {"participant_": participant, "time_": time_state},
                                        fields={simulation: ("-", "operation_active_", "bool"),
                                                time_state: ("+", "accepted_time", "double")}),
        wcsph: check_collaboration(wcsph_page, definitions[wcsph], compounds, wcsph,
                                   {"budget_": "nexora::detail::ResourceBudget",
                                    "initial_": "nexora::sph::ParticleFields3D"},
                                   bases=(participant,), fields=wcsph_fields),
    }
    leaf = Page(docs / "html" / f"{compounds[time_state]}.html")
    require(not leaf.diagram_order and not leaf.embedded and "Collaboration graph" not in leaf.images.values()
            and "Collaboration diagram" not in leaf.text and UML_LEGEND_LINK not in leaf.links,
            f"{leaf.path}: leaf class has an empty collaboration section")
    return {"index": str(home.path), "inheritance_images": inheritance_images, "private_attribute_count": len(xml_anchors),
            "private_attributes": member_types, "collaboration": collaboration, "leaf": str(leaf.path)}


def check_bad_docs(work, log, doxygen, generated_config):
    cases = {
        "bad-param": ("/// @param wrong_parameter A mismatched parameter.\nvoid sample(int actual);\n",
                      r"^.*sample\.hpp:\d+:.*argument 'wrong_parameter'.*@param.*not found.*$"),
        "bad-ref": ("/// See @ref missing_docs_reference.\nvoid sample();\n",
                    r"^.*sample\.hpp:\d+:.*unable to resolve reference.*missing_docs_reference.*$"),
    }
    results = {}
    for name, (source, diagnostic) in cases.items():
        case = work / name
        case.mkdir()
        (case / "sample.hpp").write_text(source, encoding="utf-8")
        config = case / "Doxyfile"
        config.write_text(f'@INCLUDE = "{generated_config.as_posix()}"\n'
                          f'INPUT = "{(case / "sample.hpp").as_posix()}"\n'
                          f'OUTPUT_DIRECTORY = "{(case / "output").as_posix()}"\n'
                          'USE_MDFILE_AS_MAINPAGE =\n', encoding="utf-8")
        results[name] = run([doxygen, config], log, diagnostic=diagnostic)
    return results


def validate(args):
    work = args.work_dir.resolve()
    work.mkdir(parents=True)
    logs = args.logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / "docs-validation.log"
    evidence = {"started": now(), "system": platform.system(), "architecture": platform.machine(),
                "network_isolation_required": args.require_network_blocked}
    try:
        if args.require_network_blocked:
            evidence["network_before"], blocked = verify_network()
            require(blocked, "documentation validation requires a blocked network")
        cc, cxx = (executable(name) for name in (args.cc, args.cxx))
        dot = executable(args.dot)
        evidence["versions"] = {name: run([tool, "--version"], log) for name, tool in
                                (("cc", cc), ("cxx", cxx), ("cmake", "cmake"), ("ninja", "ninja"))}
        options = [*COMMON_OPTIONS, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_COMPILER={cc}",
                   f"-DCMAKE_CXX_COMPILER={cxx}", f"-DKokkos_DIR={args.kokkos_prefix.resolve() / PACKAGE_DIR}",
                   "-DBUILD_TESTING=OFF", "-DNEXORA_ENABLE_MPI=OFF", "-DNEXORA_BUILD_DOCS=ON",
                   f"-DNEXORA_GRAPHVIZ_DOT_EXECUTABLE={dot}"]
        evidence["graphviz"] = {"path": str(dot), "version": run([dot, "-V"], log)}
        if args.doxygen:
            options.append(f"-DDOXYGEN_EXECUTABLE={executable(args.doxygen)}")
        for precision in ("float", "double"):
            build = work / f"docs-{precision}"
            home = build / "docs/html/index.html"
            internal_home = build / "docs-internal/html/index.html"
            double = "ON" if precision == "double" else "OFF"
            run(["cmake", "-S", ROOT, "-B", build, *options,
                 f"-DNEXORA_ENABLE_DOUBLE_PRECISION={double}"], log)
            require(not home.exists(), f"{home} must be absent after configuration")
            require(not internal_home.exists(), f"{internal_home} must be absent after configuration")
            if precision == "float":
                run(["cmake", "--build", build], log)
                require(not home.exists(), f"{home} must be absent after the ordinary build")
                require(not internal_home.exists(), f"{internal_home} must be absent after the ordinary build")
            run(["cmake", "--build", build, "--target", "nexora_docs"], log)
            require(not internal_home.exists(), f"{internal_home} must be absent after building nexora_docs")
            evidence[precision] = {"index_exists_after_configure": False, **check_docs(build, precision)}
            if precision == "float":
                evidence[precision]["index_exists_after_ordinary_build"] = False

        build = work / "docs-float"
        run(["cmake", "--build", build, "--target", "nexora_internal_docs"], log)
        evidence["internal"] = check_internal_docs(build)
        check_docs(build, "float")
        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
        match = re.search(r"^DOXYGEN_EXECUTABLE:[^=]+=(.+)$", cache, re.MULTILINE)
        require(match, f"{build}: FindDoxygen did not record its executable")
        doxygen = match.group(1)
        evidence["doxygen"] = {"path": doxygen, "version": run([doxygen, "--version"], log)}
        evidence["warning_failures"] = check_bad_docs(work, log, doxygen, build / "Doxyfile.nexora_docs")
        if args.require_network_blocked:
            evidence["network_after"], blocked = verify_network()
            require(blocked, "network isolation was lost during documentation validation")
    except (OSError, RuntimeError, ET.ParseError) as error:
        evidence["failure"] = str(error)
        raise
    finally:
        evidence["finished"] = now()
        write_json(logs / "docs-validation.json", evidence)
    print(f"Validated both API references, the internal manual and both documentation diagnostics. Evidence: {logs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kokkos-prefix", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="new directory for the documentation builds")
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--doxygen", help="explicit tool path for FindDoxygen")
    parser.add_argument("--dot", default="dot", help="Graphviz dot executable")
    parser.add_argument("--require-network-blocked", action="store_true")
    try:
        validate(parser.parse_args())
    except (OSError, RuntimeError, ET.ParseError) as error:
        print(f"Documentation validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
