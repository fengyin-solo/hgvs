"""Tests for hgvs.audit round-trip auditing.

These tests are purely local: auditing must work from Parser + format
alone, without a UTA/SeqRepo data provider or normalization.
"""

import copy
import json

import pytest

import hgvs
import hgvs.audit
import hgvs.edit
import hgvs.location
import hgvs.posedit
import hgvs.sequencevariant
from hgvs.parsers import PYPARSING_GRAMMAR, Parser

parser = Parser()
pyparsing_parser = Parser(grammar_fn=PYPARSING_GRAMMAR)


def _audit(value, conf=None, p=None):
    return hgvs.audit.audit_roundtrip(value, conf, parser=p or parser)


def _assert_serializable(report):
    data = report.to_dict()
    json.dumps(data)  # must raise if any value were non-primitive
    assert data["input_kind"] in ("string", "variant")
    assert isinstance(data["difference_categories"], list)
    assert isinstance(data["differences"], list)
    return data


# -- stable round trips ------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize(
    "hgvs_text",
    [
        "NM_000007.13:g.36561662A>C",
        "NM_01234.5:c.22+1A>T",
        "NM_01234.5:c.87+1_88-1del",
        "NM_01234.5:c.*13A>C",
        "NM_01234.5:c.-10_-5dup",
        "NM_000007.13:g.123_124inv",
        "NM_000007.13:g.123_124conNM_000008.1:g.124_125",
        "NP_001628.1:p.Gly528Arg",
        "NP_001628.1:p.Arg96fsTer17",
        "NP_001628.1:p.Ter410GlnextTer17",
        "NP_001628.1:p.Gly528=",
        "NP_001628.1:p.?",
        "NM_01234.5(BOGUS):c.65A>C",
        "NC_000009.11:g.(?_108337304)_(108337428_?)del",
        "NC_000005.9:g.(90136803_90159675)dup",
    ],
)
def test_stable_roundtrips(hgvs_text):
    report = _audit(hgvs_text)
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_NONE
    assert report.canonical_text == hgvs_text
    assert report.error_stage is None
    assert report.variant_type == hgvs_text.split(":")[1].split(".")[0]
    _assert_serializable(report)


# -- reference sequence omission ---------------------------------------------


@pytest.mark.quick
def test_deleted_reference_omitted_by_default_config():
    report = _audit("NM_001166478.1:c.31_32delTT")
    assert report.canonical_text == "NM_001166478.1:c.31_32del"
    assert report.semantic_equivalent is False
    assert report.difference_category == hgvs.audit.CATEGORY_EDIT_REFERENCE_OMITTED
    assert report.error_stage is None
    (diff,) = report.differences
    assert diff["path"] == "variant.posedit.edit.ref"
    assert diff["kind"] == hgvs.audit.CATEGORY_EDIT_REFERENCE_OMITTED
    assert diff["original"] == "TT"
    assert not diff["canonical"]  # re-parsed "del" has no reference (""/None)
    _assert_serializable(report)


@pytest.mark.quick
def test_duplicated_reference_omitted_by_default_config():
    report = _audit("NM_001166478.1:c.35_36dupTC")
    assert report.canonical_text == "NM_001166478.1:c.35_36dup"
    assert report.semantic_equivalent is False
    assert report.difference_category == hgvs.audit.CATEGORY_EDIT_REFERENCE_OMITTED
    assert report.differences[0]["original"] == "TC"


@pytest.mark.quick
def test_identity_reference_omitted_by_default_config():
    report = _audit("NM_001166478.1:c.31T=")
    assert report.canonical_text == "NM_001166478.1:c.31="
    assert report.semantic_equivalent is False
    assert report.difference_category == hgvs.audit.CATEGORY_EDIT_REFERENCE_OMITTED
    paths = {d["path"] for d in report.differences}
    # identity folds ref/alt into "="; only the reference side is reported
    assert paths == {"variant.posedit.edit.ref"}


@pytest.mark.quick
@pytest.mark.parametrize(
    ("hgvs_text", "conf"),
    [
        ("NM_001166478.1:c.31_32delTT", {"max_ref_length": None}),
        ("NM_001166478.1:c.31_32del2", {"max_ref_length": None}),
        ("NM_001166478.1:c.31_32delTT", {"max_ref_length": 2}),
        ("NM_001166478.1:c.31T=", {"max_ref_length": None}),
    ],
)
def test_reference_preserved_with_matching_config(hgvs_text, conf):
    report = _audit(hgvs_text, conf)
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_NONE
    assert report.canonical_text == hgvs_text


@pytest.mark.quick
def test_reference_accession_omitted_variant_object():
    variant = hgvs.sequencevariant.SequenceVariant(
        ac=None,
        type="c",
        posedit=hgvs.posedit.PosEdit(
            pos=hgvs.location.Interval(
                start=hgvs.location.SimplePosition(31),
                end=hgvs.location.SimplePosition(32),
            ),
            edit=hgvs.edit.NARefAlt(ref="TT", alt=None),
        ),
    )
    report = _audit(variant, {"max_ref_length": None})
    assert report.input_kind == "variant"
    # original_text uses global settings (max_ref_length=0 hides TT);
    # conf only changes the canonical display
    assert report.original_text == "c.31_32del"
    assert report.canonical_text == "c.31_32delTT"
    assert report.semantic_equivalent is False
    assert report.error_stage == hgvs.audit.STAGE_REPARSE
    assert report.difference_category == hgvs.audit.CATEGORY_REFERENCE_ACCESSION_OMITTED
    assert report.error_type == "HGVSParseError"
    _assert_serializable(report)


@pytest.mark.quick
def test_bare_reference_less_string_fails_at_parse_stage():
    report = _audit("g.36561662A>C")
    assert report.error_stage == hgvs.audit.STAGE_PARSE
    assert report.difference_category == hgvs.audit.CATEGORY_PARSE_ERROR
    assert report.canonical_text is None
    assert report.variant_type is None
    _assert_serializable(report)


# -- amino acid letter code and terminator notation --------------------------


@pytest.mark.quick
def test_one_letter_amino_acid_display_config():
    report = _audit("NP_001628.1:p.Gly528Arg", {"p_3_letter": False})
    assert report.canonical_text == "NP_001628.1:p.G528R"
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_AMINO_ACID_LETTER_CODE
    assert report.differences == ()


@pytest.mark.quick
def test_one_letter_amino_acid_input_normalized_to_three_letter():
    report = _audit("NP_001628.1:p.G528R")
    assert report.canonical_text == "NP_001628.1:p.Gly528Arg"
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_AMINO_ACID_LETTER_CODE


@pytest.mark.quick
def test_terminator_asterisk_display_config():
    report = _audit("NP_001628.1:p.Gly528Ter", {"p_term_asterisk": True})
    assert report.canonical_text == "NP_001628.1:p.Gly528*"
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_TERMINATOR_SYMBOL


@pytest.mark.quick
def test_combined_letter_and_terminator_attributed_to_letter_dimension():
    # At one-letter mode "*" is the one-letter Ter, not a separate choice
    report = _audit("NP_001628.1:p.G528*")
    assert report.canonical_text == "NP_001628.1:p.Gly528Ter"
    assert report.semantic_equivalent is True
    assert report.difference_categories == (hgvs.audit.CATEGORY_AMINO_ACID_LETTER_CODE,)


@pytest.mark.quick
def test_frameshift_terminator_display_config():
    report = _audit("NP_001628.1:p.Arg96fsTer17", {"p_term_asterisk": True})
    assert report.canonical_text == "NP_001628.1:p.Arg96fs*17"
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_TERMINATOR_SYMBOL


# -- uncertainty and intron offsets ------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize(
    "hgvs_text",
    [
        "NP_001628.1:p.(Gly528Arg)",
        "NC_000009.11:g.(?_108337304)_(108337428_?)del",
        "NC_000005.9:g.(90136803_90159675)dup",
    ],
)
def test_uncertainty_roundtrips(hgvs_text):
    report = _audit(hgvs_text)
    assert report.semantic_equivalent is True
    assert report.canonical_text == hgvs_text
    assert report.difference_category == hgvs.audit.CATEGORY_NONE


@pytest.mark.quick
def test_intron_offsets_and_datum_preserved():
    report = _audit("NM_01234.5:c.87+1_88-1del")
    assert report.semantic_equivalent is True
    variant = parser.parse("NM_01234.5:c.87+1_88-1del")
    start = variant.posedit.pos.start
    assert start.base == 87
    assert start.offset == 1


@pytest.mark.quick
def test_semantic_differences_classify_fields():
    original = parser.parse("NM_01234.5:c.87+1A>T")

    intron_variant = copy.deepcopy(original)
    intron_variant.posedit.pos.start.offset = 2
    kinds = {d["kind"] for d in hgvs.audit.semantic_differences(original, intron_variant)}
    assert kinds == {hgvs.audit.CATEGORY_INTRON_OFFSET}

    coord_variant = copy.deepcopy(original)
    coord_variant.posedit.pos.start.base = 88
    kinds = {d["kind"] for d in hgvs.audit.semantic_differences(original, coord_variant)}
    assert kinds == {hgvs.audit.CATEGORY_POSITION_COORDINATE}

    uncertain_variant = copy.deepcopy(original)
    uncertain_variant.posedit.uncertain = True
    kinds = {d["kind"] for d in hgvs.audit.semantic_differences(original, uncertain_variant)}
    assert kinds == {hgvs.audit.CATEGORY_UNCERTAINTY}

    datum_variant = parser.parse("NM_01234.5:c.*13A>C")
    assert (
        hgvs.audit.semantic_differences(parser.parse("NM_01234.5:c.13A>C"), datum_variant)[0][
            "kind"
        ]
        == hgvs.audit.CATEGORY_POSITION_DATUM
    )


# -- configuration handling ---------------------------------------------------


@pytest.mark.quick
def test_conf_does_not_mutate_global_config():
    before = {
        key: getattr(hgvs.global_config.formatting, key)
        for key in ("max_ref_length", "p_3_letter", "p_term_asterisk", "p_init_met")
    }
    _audit("NM_001166478.1:c.31_32delTT", {"max_ref_length": None})
    _audit("NP_001628.1:p.Gly528Ter", {"p_term_asterisk": True, "p_3_letter": False})
    after = {
        key: getattr(hgvs.global_config.formatting, key)
        for key in ("max_ref_length", "p_3_letter", "p_term_asterisk", "p_init_met")
    }
    assert before == after


@pytest.mark.quick
def test_conf_affects_display_only_not_input_object():
    variant = parser.parse("NM_001166478.1:c.31_32delTT")
    snapshot = repr(variant)
    report = _audit(variant, {"max_ref_length": None})
    assert report.semantic_equivalent is True
    assert variant.posedit.edit.ref == "TT"
    assert repr(variant) == snapshot
    # no normalization happened: type and coordinates are exactly as parsed
    assert variant.posedit.pos.start.base == 31


@pytest.mark.quick
def test_distinct_configs_distinguish_omission_from_preservation():
    text = "NM_001166478.1:c.31_32delTT"
    truncated = _audit(text)  # global default max_ref_length = 0
    bounded = _audit(text, {"max_ref_length": 1})
    preserved = _audit(text, {"max_ref_length": None})
    assert truncated.canonical_text.endswith("del")
    assert bounded.canonical_text.endswith("del")
    assert preserved.canonical_text.endswith("delTT")
    assert truncated.semantic_equivalent is False
    assert bounded.semantic_equivalent is False
    assert preserved.semantic_equivalent is True


# -- error inputs -------------------------------------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("bad_text", ["not an hgvs", "", "NM_000001.2:z.1A>T", "c.[1A>T;2A>T]"])
def test_unparseable_inputs_report_parse_stage(bad_text):
    report = _audit(bad_text)
    assert report.error_stage == hgvs.audit.STAGE_PARSE
    assert report.difference_category == hgvs.audit.CATEGORY_PARSE_ERROR
    assert report.original_text == bad_text
    assert report.semantic_equivalent is False
    assert report.error_type
    _assert_serializable(report)


@pytest.mark.quick
def test_format_failure_reported_at_format_stage():
    variant = hgvs.sequencevariant.SequenceVariant(
        ac="NM_000007.13",
        type="g",
        posedit=hgvs.posedit.PosEdit(
            pos=hgvs.location.Interval(
                start=hgvs.location.SimplePosition(123),
                end=hgvs.location.SimplePosition(124),
            ),
            edit=hgvs.edit.NARefAlt(ref=None, alt=None),
        ),
    )
    report = _audit(variant)
    assert report.error_stage == hgvs.audit.STAGE_FORMAT
    assert report.difference_category == hgvs.audit.CATEGORY_FORMAT_ERROR
    assert report.canonical_text is None
    assert report.variant_type == "g"
    assert report.error_type == "HGVSError"
    _assert_serializable(report)


@pytest.mark.quick
def test_reparse_failure_when_accession_present_is_generic():
    variant = hgvs.sequencevariant.SequenceVariant(ac="NM_000001.2", type="c", posedit=None)
    report = _audit(variant)
    assert report.canonical_text == "NM_000001.2:c.?"
    assert report.error_stage == hgvs.audit.STAGE_REPARSE
    assert report.difference_category == hgvs.audit.CATEGORY_REPARSE_ERROR
    _assert_serializable(report)


@pytest.mark.quick
def test_invalid_input_type_raises_type_error():
    with pytest.raises(TypeError):
        hgvs.audit.audit_roundtrip(12345, parser=parser)


# -- variant-object input, stability, and parser injection --------------------


@pytest.mark.quick
def test_variant_object_input_matches_string_input():
    variant = parser.parse("NM_000007.13:g.36561662A>C")
    object_report = _audit(variant)
    string_report = _audit("NM_000007.13:g.36561662A>C")
    assert object_report.semantic_equivalent == string_report.semantic_equivalent
    assert object_report.canonical_text == string_report.canonical_text
    assert object_report.difference_category == string_report.difference_category
    assert object_report.input_kind == "variant"
    assert string_report.input_kind == "string"


@pytest.mark.quick
def test_repeated_audits_are_stable_and_non_mutating():
    text = "NM_001166478.1:c.31_32delTT"
    first = _audit(text).to_dict()
    second = _audit(text).to_dict()
    assert first == second

    variant = parser.parse("NP_001628.1:p.Gly528Ter")
    variant_snapshot = repr(variant)
    conf = {"p_term_asterisk": True}
    r1 = _audit(variant, conf).to_dict()
    r2 = _audit(variant, conf).to_dict()
    assert r1 == r2
    assert repr(variant) == variant_snapshot

    # auditing the canonical output again stays stable
    canonical = r1["canonical_text"]
    again = _audit(canonical, conf)
    assert again.canonical_text == canonical
    assert again.semantic_equivalent is True


@pytest.mark.quick
def test_audit_works_with_injected_pyparsing_parser():
    text = "NM_001166478.1:c.31_32delTT"
    report = _audit(text, p=pyparsing_parser)
    assert report.difference_category == hgvs.audit.CATEGORY_EDIT_REFERENCE_OMITTED
    assert report.error_stage is None

    bad = _audit("not an hgvs", p=pyparsing_parser)
    assert bad.error_stage == hgvs.audit.STAGE_PARSE


@pytest.mark.quick
def test_leading_whitespace_is_cosmetic_only():
    # the OMeta parser tolerates leading whitespace; the object graph is intact
    report = _audit(" NP_001628.1:p.Gly528Arg")
    assert report.semantic_equivalent is True
    assert report.difference_category == hgvs.audit.CATEGORY_OTHER_REPRESENTATION
