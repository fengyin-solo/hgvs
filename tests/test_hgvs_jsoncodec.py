"""Tests for the versioned, structured JSON codec (hgvs.jsoncodec)"""

import copy
import json

import pytest

import hgvs.edit
import hgvs.location
import hgvs.posedit
import hgvs.sequencevariant
from hgvs import jsoncodec
from hgvs.enums import Datum
from hgvs.exceptions import HGVSJSONError
from hgvs.hgvsposition import HGVSPosition
from hgvs.parsers import Parser

# Format with the full reference sequence, so that round-tripping does
# not depend on the global max_ref_length display truncation.
FULL_REF_CONF = {"max_ref_length": None}

parser = Parser()


# ---------------------------------------------------------------------------
# Structural round-trip: formatted strings and concrete object types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hgvs_text",
    [
        # nucleic acid substitutions, indels, counted deletions, dup, inv
        "NC_000007.13:g.36561662A>G",
        "NC_000007.13:g.36561662_36561663insT",
        "NM_001166478.1:c.31_32del2",
        "NM_001166478.1:c.31_32delTT",
        "NM_001166478.1:c.2_7delinsTTTAGA",
        "NM_001166478.1:c.35_36dupTC",
        "NM_001166478.1:c.31_32inv",
        "NM_001166478.1:c.31_32delTTinsAA",
        # identity (ref == alt) must stay an NARefAlt, not become a generic string
        "NM_001166478.1:c.31=",
        # UTR and intronic coordinates preserve datum and offset
        "NM_001166478.1:c.-56G>A",
        "NM_001166478.1:c.*4C>T",
        "NM_001166478.1:c.1582+12_1582+14del",
        "NM_001166478.1:c.1582-2A>G",
        # SEQ_START datum (n. coordinates)
        "NM_001166478.1:n.35del",
        # gene symbol and accession
        "NM_01234.5(BOGUS):c.65A>C",
        # conversion keeps its nested position structure
        "NC_000007.13:g.117199563_117199565conNG_007466.2:g.1234_1236",
        # Invitae-style copy number edit
        "NC_000007.13:g.1copy3",
        # protein variants: substitution (AASub), frameshift, delins, special forms
        "NP_001628.1:p.Gly528Arg",
        "NP_001628.1:p.Glu5ValfsTer5",
        "NP_001628.1:p.Cys28_Met30delinsSerGlu",
        "NP_001628.1:p.(Gly528Arg)",
        "NP_001628.1:p.?",
        "NP_001628.1:p.0",
        "NP_001628.1:p.0?",
        "NP_001628.1:p.=",
        "NP_001628.1:p.(=)",
        # protein extension
        "NP_000040.1:p.Met1ext-12",
    ],
)
def test_roundtrip_text(hgvs_text):
    var = parser.parse(hgvs_text)
    decoded = jsoncodec.loads(jsoncodec.dumps(var))

    assert isinstance(decoded, hgvs.sequencevariant.SequenceVariant)
    # formatting with full references reproduces the original object's format
    assert decoded.format(FULL_REF_CONF) == var.format(FULL_REF_CONF)
    # and the default (truncated) formatting agrees as well
    assert str(decoded) == str(var)


@pytest.mark.parametrize(
    "hgvs_text",
    [
        # whole interval uncertain, endpoints certain
        "NC_000005.9:g.(90136803_90159675)dup",
        # each side of the outer interval is a nested uncertain interval
        "NC_000005.9:g.(90136803_90144453)_(90159675_90261231)dup",
        # unknown breakpoints inside nested intervals
        "NC_000009.11:g.(?_108337304)_(108337428_?)del",
        # nested uncertain intervals on CDS coordinates, with intronic offsets
        "NM_004484.3:c.(1293-76_1293)_(1413_1413+55)del",
        # individually uncertain endpoint
        "NM_001166478.1:c.(407_410)del",
        # uncertain protein posedit
        "NP_001628.1:p.(Gly528Arg)",
    ],
)
def test_roundtrip_uncertain(hgvs_text):
    var = parser.parse(hgvs_text)
    decoded = jsoncodec.loads(jsoncodec.dumps(var))

    assert decoded.format(FULL_REF_CONF) == var.format(FULL_REF_CONF)
    # uncertainty structure survives: compare the encoded documents,
    # which encode every uncertain flag explicitly
    assert jsoncodec.to_document(decoded) == jsoncodec.to_document(var)


def test_uncertain_interval_structure_is_nested():
    var = parser.parse("NC_000009.11:g.(?_108337304)_(108337428_?)del")
    decoded = jsoncodec.loads(jsoncodec.dumps(var))

    pos = decoded.posedit.pos
    assert isinstance(pos, hgvs.location.Interval)
    # endpoints remain intervals (they are not flattened into points)
    assert isinstance(pos.start, hgvs.location.Interval)
    assert isinstance(pos.end, hgvs.location.Interval)
    assert pos.start.uncertain is True
    assert pos.end.uncertain is True
    assert pos.start.start.base is None
    assert pos.start.end.base == 108337304
    assert pos.end.start.base == 108337428
    assert pos.end.end.base is None


def test_cds_uncertain_interval_keeps_datum_and_offsets():
    var = parser.parse("NM_004484.3:c.(1293-76_1293)_(1413_1413+55)del")
    decoded = jsoncodec.loads(jsoncodec.dumps(var))

    pos = decoded.posedit.pos
    assert isinstance(pos, hgvs.location.BaseOffsetInterval)
    assert isinstance(pos.start, hgvs.location.BaseOffsetInterval)
    assert isinstance(pos.end, hgvs.location.BaseOffsetInterval)
    start_bp = pos.start.start
    end_bp = pos.end.end
    assert isinstance(start_bp, hgvs.location.BaseOffsetPosition)
    assert start_bp.base == 1293
    assert start_bp.offset == -76
    assert start_bp.datum == Datum.CDS_START
    assert end_bp.base == 1413
    assert end_bp.offset == 55
    # the edit type is preserved
    assert isinstance(decoded.posedit.edit, hgvs.edit.NARefAlt)
    assert decoded.posedit.edit.type == "del"


def test_core_fields_preserved():
    var = parser.parse("NM_01234.5(BOGUS):c.*4C>T")
    decoded = jsoncodec.loads(jsoncodec.dumps(var))

    assert decoded.ac == "NM_01234.5"
    assert decoded.type == "c"
    assert decoded.gene == "BOGUS"
    bp = decoded.posedit.pos.start
    assert bp.base == 4
    assert bp.offset == 0
    assert bp.datum == Datum.CDS_END
    edit = decoded.posedit.edit
    assert isinstance(edit, hgvs.edit.NARefAlt)
    assert (edit.ref, edit.alt, edit.type) == ("C", "T", "sub")
    assert decoded.posedit.uncertain is False


def test_protein_types_are_not_flattened():
    sub = parser.parse("NP_001628.1:p.Gly528Arg")
    ext = parser.parse("NP_000040.1:p.Met1ext-12")
    fs = parser.parse("NP_001628.1:p.Glu5ValfsTer5")
    delins = parser.parse("NP_001628.1:p.Cys28_Met30delinsSerGlu")

    sub2 = jsoncodec.loads(jsoncodec.dumps(sub))
    ext2 = jsoncodec.loads(jsoncodec.dumps(ext))
    fs2 = jsoncodec.loads(jsoncodec.dumps(fs))
    delins2 = jsoncodec.loads(jsoncodec.dumps(delins))

    assert isinstance(sub2.posedit.edit, hgvs.edit.AASub)
    assert isinstance(ext2.posedit.edit, hgvs.edit.AAExt)
    assert ext2.posedit.edit.length == -12
    assert isinstance(fs2.posedit.edit, hgvs.edit.AAFs)
    assert fs2.posedit.edit.length == 5
    assert isinstance(delins2.posedit.edit, hgvs.edit.AARefAlt)
    assert delins2.posedit.edit.type == "delins"
    # the position carries the reference residue/coordinate; the
    # alternate residue is in the AASub edit
    assert sub2.posedit.pos.start.aa == "G"
    assert sub2.posedit.pos.start.base == 528
    assert sub2.posedit.edit.alt == "R"
    for before, after in ((sub, sub2), (ext, ext2), (fs, fs2), (delins, delins2)):
        assert str(after) == str(before)


def test_repeat_edit_type_preserved():
    # the HGVS parser does not read repeat syntax yet, so build directly
    var = parser.parse("NM_001166478.1:c.30_32del")
    var.posedit.edit = hgvs.edit.Repeat(ref="CAG", min=12, max=34)

    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert isinstance(decoded.posedit.edit, hgvs.edit.Repeat)
    assert decoded.posedit.edit.type == "repeat"
    assert (decoded.posedit.edit.ref, decoded.posedit.edit.min, decoded.posedit.edit.max) == (
        "CAG",
        12,
        34,
    )
    assert (
        decoded.format(FULL_REF_CONF)
        == var.format(FULL_REF_CONF)
        == ("NM_001166478.1:c.30_32CAG(12_34)")
    )

    # exact-count repeat form
    var.posedit.edit = hgvs.edit.Repeat(ref="C", min=3, max=3)
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert isinstance(decoded.posedit.edit, hgvs.edit.Repeat)
    assert decoded.format(FULL_REF_CONF) == "NM_001166478.1:c.30_32C[3]"


def test_extension_edit_fields():
    ext = hgvs.edit.AAExt(ref="", alt=None, aaterm=None, length=-5)
    var = hgvs.sequencevariant.SequenceVariant(
        ac="NP_1.1",
        type="p",
        posedit=hgvs.posedit.PosEdit(
            pos=hgvs.location.Interval(
                start=hgvs.location.AAPosition(base=1, aa="M"),
                end=hgvs.location.AAPosition(base=1, aa="M"),
            ),
            edit=ext,
        ),
    )
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert isinstance(decoded.posedit.edit, hgvs.edit.AAExt)
    assert decoded.posedit.edit.type == "ext"
    assert decoded.posedit.edit.length == -5
    assert str(decoded) == str(var)


def test_identity_edits_keep_class():
    # nucleic acid identity (T=)
    na = hgvs.edit.NARefAlt(ref="T", alt="T")
    assert na.type == "identity"
    decoded = jsoncodec.loads(jsoncodec.dumps(na))
    assert isinstance(decoded, hgvs.edit.NARefAlt)
    assert decoded.type == "identity"

    # protein identity (Gly=) as an edit object (p.= text uses the literal form)
    aa = hgvs.edit.AARefAlt(ref="G", alt="G")
    assert aa.type == "identity"
    decoded = jsoncodec.loads(jsoncodec.dumps(aa))
    assert isinstance(decoded, hgvs.edit.AARefAlt)
    assert not isinstance(decoded, hgvs.edit.AASub)
    assert decoded.type == "identity"


def test_uncertain_edit_flag_preserved():
    edit = hgvs.edit.NARefAlt(ref="A", alt="T", uncertain=True)
    decoded = jsoncodec.loads(jsoncodec.dumps(edit))
    assert isinstance(decoded, hgvs.edit.NARefAlt)
    assert decoded.uncertain is True
    assert str(decoded) == "(A>T)"


def test_intronic_offset_none_roundtrips():
    # "+?" form: offset is unknown (None)
    pos = hgvs.location.BaseOffsetPosition(base=10, offset=None, datum=Datum.CDS_START)
    var = hgvs.sequencevariant.SequenceVariant(
        ac="NM_1.1",
        type="c",
        posedit=hgvs.posedit.PosEdit(
            pos=hgvs.location.Interval(start=pos, end=pos),
            edit=hgvs.edit.NARefAlt(ref="A", alt="T"),
        ),
    )
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert decoded.posedit.pos.start.offset is None
    assert str(decoded) == "NM_1.1:c.10+?A>T"


def test_simple_position_uncertain_and_unknown_base():
    pos = hgvs.location.SimplePosition(base=None, uncertain=True)
    decoded = jsoncodec.loads(jsoncodec.dumps(pos))
    assert isinstance(decoded, hgvs.location.SimplePosition)
    assert decoded.base is None
    assert decoded.uncertain is True
    assert str(decoded) == "(?)"


def test_posedit_and_variant_none_roundtrip():
    # p.? parses to a variant with posedit=None entirely
    var = parser.parse("NP_001628.1:p.?")
    assert var.posedit is None
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert decoded.posedit is None
    assert str(decoded) == "NP_001628.1:p.?"

    # the p.0 form instead keeps a PosEdit with a literal "0" edit
    var_zero = parser.parse("NP_001628.1:p.0")
    decoded_zero = jsoncodec.loads(jsoncodec.dumps(var_zero))
    assert decoded_zero.posedit is not None
    assert decoded_zero.posedit.pos is None
    assert decoded_zero.posedit.edit == "0"
    assert str(decoded_zero) == "NP_001628.1:p.0"

    # a variant constructed without a posedit
    var = hgvs.sequencevariant.SequenceVariant(ac="NM_1.1", type="c", posedit=None)
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert decoded.posedit is None
    assert str(decoded) == "NM_1.1:c.?"


def test_hgvs_position_roundtrip():
    pos = parser.parse_hgvs_position("AC_01234.5:g.1234")
    assert isinstance(pos, HGVSPosition)
    decoded = jsoncodec.loads(jsoncodec.dumps(pos))
    assert isinstance(decoded, HGVSPosition)
    assert decoded.ac == "AC_01234.5"
    assert decoded.type == "g"
    assert str(decoded) == "AC_01234.5:g.1234"

    pos_p = parser.parse_hgvs_position("AC_01234.5:p.Met1_Gly10")
    decoded_p = jsoncodec.loads(jsoncodec.dumps(pos_p))
    assert isinstance(decoded_p.pos, hgvs.location.Interval)
    assert isinstance(decoded_p.pos.start, hgvs.location.AAPosition)
    assert str(decoded_p) == "AC_01234.5:p.Met1_Gly10"


# ---------------------------------------------------------------------------
# Envelope / schema
# ---------------------------------------------------------------------------


def test_envelope_has_version_and_type():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)

    assert set(document) == {"schema", "schema_version", "object_type", "data"}
    assert document["schema"] == jsoncodec.SCHEMA_NAME
    assert document["schema_version"] == jsoncodec.SCHEMA_VERSION
    assert document["object_type"] == "SequenceVariant"
    assert document["data"]["object_type"] == "SequenceVariant"
    # every nested object carries an explicit type tag
    edit_node = document["data"]["posedit"]["edit"]
    assert edit_node["object_type"] == "NARefAlt"


def test_dumps_is_valid_json():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = json.loads(jsoncodec.dumps(var))
    assert document["object_type"] == "SequenceVariant"


def test_unknown_schema_version_rejected():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)

    for bad_version in (2, 0, -1, "1", "1.0", None, 1.0):
        bad = copy.deepcopy(document)
        bad["schema_version"] = bad_version
        with pytest.raises(HGVSJSONError, match="schema version"):
            jsoncodec.from_document(bad)


def test_unknown_schema_name_rejected():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)
    document["schema"] = "something-else"
    with pytest.raises(HGVSJSONError, match="unknown schema"):
        jsoncodec.from_document(document)


def test_unknown_object_type_rejected():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)
    document["object_type"] = "Bogus"
    document["data"]["object_type"] = "Bogus"
    with pytest.raises(HGVSJSONError, match="unknown object_type"):
        jsoncodec.from_document(document)

    # unknown nested type
    document = jsoncodec.to_document(var)
    document["data"]["posedit"]["edit"]["object_type"] = "BogusEdit"
    with pytest.raises(HGVSJSONError, match="unknown object_type"):
        jsoncodec.from_document(document)


def test_envelope_data_type_mismatch_rejected():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)
    document["object_type"] = "PosEdit"
    with pytest.raises(HGVSJSONError, match="does not match"):
        jsoncodec.from_document(document)


@pytest.mark.parametrize(
    "payload",
    [
        "{not valid json",
        "",
        "[]",
        "123",
        '"a string"',
        "null",
        "{}",
        # well-formed JSON but structurally wrong
        '{"schema": "hgvs-variant-json", "schema_version": 1, "object_type": "SequenceVariant"}',
    ],
)
def test_corrupt_payload_rejected(payload):
    with pytest.raises(HGVSJSONError):
        jsoncodec.loads(payload)


def test_non_dict_loaded_value_rejected():
    with pytest.raises(HGVSJSONError):
        jsoncodec.loads(123)


def test_missing_and_mistyped_fields_rejected():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    base = jsoncodec.to_document(var)

    missing = copy.deepcopy(base)
    del missing["data"]["posedit"]["edit"]["ref"]
    with pytest.raises(HGVSJSONError, match="missing required field"):
        jsoncodec.from_document(missing)

    mistyped = copy.deepcopy(base)
    mistyped["data"]["posedit"]["edit"]["uncertain"] = "yes"
    with pytest.raises(HGVSJSONError, match="expected a boolean"):
        jsoncodec.from_document(mistyped)

    bad_base = copy.deepcopy(base)
    bad_base["data"]["posedit"]["pos"]["start"]["base"] = "31"
    with pytest.raises(HGVSJSONError, match="expected an integer"):
        jsoncodec.from_document(bad_base)

    bad_datum = copy.deepcopy(base)
    bad_datum["data"]["posedit"]["pos"]["start"]["datum"] = "BOGUS"
    with pytest.raises(HGVSJSONError, match="unknown datum"):
        jsoncodec.from_document(bad_datum)


def test_strict_mode_rejects_extra_fields():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)

    # extra nested field
    bad = copy.deepcopy(document)
    bad["data"]["posedit"]["edit"]["comment"] = "unexpected"
    with pytest.raises(HGVSJSONError, match="unexpected field"):
        jsoncodec.from_document(bad)

    # extra envelope field
    bad = copy.deepcopy(document)
    bad["meta"] = {}
    with pytest.raises(HGVSJSONError, match="unexpected top-level field"):
        jsoncodec.from_document(bad)


def test_lenient_mode_ignores_extra_fields():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    document = jsoncodec.to_document(var)
    document["data"]["posedit"]["edit"]["comment"] = "ignored"
    document["meta"] = {"source": "test"}

    decoded = jsoncodec.from_document(document, strict=False)
    assert str(decoded) == str(var)


def test_encoding_unsupported_type_rejected():
    with pytest.raises(HGVSJSONError, match="not supported"):
        jsoncodec.dumps(object())


# ---------------------------------------------------------------------------
# Stability and non-mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [1, 2, 5])
def test_repeated_encode_decode_stable(rounds):
    var = parser.parse("NM_004484.3:c.(1293-76_1293)_(1413_1413+55)del")
    original_doc = jsoncodec.to_document(var)

    current = var
    for _ in range(rounds):
        current = jsoncodec.loads(jsoncodec.dumps(current))
        assert jsoncodec.to_document(current) == original_doc
        assert current.format(FULL_REF_CONF) == var.format(FULL_REF_CONF)


def test_encoding_does_not_mutate_input():
    var = parser.parse("NM_001166478.1:c.31_32del2")
    before = jsoncodec.to_document(var)
    before_text = str(var)

    jsoncodec.dumps(var)
    jsoncodec.to_document(var)

    assert jsoncodec.to_document(var) == before
    assert str(var) == before_text


def test_decoded_object_is_independent():
    var = parser.parse("NM_001166478.1:c.31_32delinsTT")
    decoded = jsoncodec.loads(jsoncodec.dumps(var))
    assert decoded is not var
    assert decoded.posedit is not var.posedit
    assert decoded.posedit.edit is not var.posedit.edit

    decoded.posedit.edit.alt = "G"
    assert var.posedit.edit.alt == "TT"
