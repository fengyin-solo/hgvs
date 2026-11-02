"""Round-trip auditing for HGVS variant strings and objects.

External laboratory data often spells the same biological variant in
more than one way.  When such a string is parsed and formatted again,
the result may differ because of an omitted reference sequence, one- or
three-letter amino acid notation, ``Ter`` versus ``*`` terminators,
uncertainty parentheses, intronic offsets, or display configuration.

This module audits that **parse -> format -> parse** round trip using
only the existing :class:`hgvs.parsers.Parser` and
:meth:`hgvs.sequencevariant.SequenceVariant.format`.  It never calls a
normalizer, a data provider, or any other external reference sequence
service, and it never modifies a variant object passed by the caller.
The *conf* argument is forwarded verbatim to ``format``; it affects
display only and keeps the package-wide precedence (per-key *conf*
override, then ``hgvs.global_config``).

The public entry point is :func:`audit_roundtrip`; the structural field
comparison used for classification is also available as
:func:`semantic_differences`.

Examples:

A stable variant round-trips without differences:

>>> import hgvs.audit
>>> report = hgvs.audit.audit_roundtrip("NM_000007.13:g.36561662A>C")
>>> report.semantic_equivalent
True
>>> report.difference_category
'none'

The default ``max_ref_length`` display setting hides deleted reference
bases, which genuinely loses information when the canonical text is
re-parsed.  This is classified from the edit fields, not from a text
diff:

>>> report = hgvs.audit.audit_roundtrip("NM_001166478.1:c.31_32delTT")
>>> report.semantic_equivalent
False
>>> report.difference_category
'edit_reference_omitted'
>>> report.error_stage is None
True

Reports always serialize to plain JSON-compatible data:

>>> import json
>>> payload = json.dumps(report.to_dict())
>>> json.loads(payload)["difference_category"]
'edit_reference_omitted'
"""

import enum
from functools import lru_cache

import attr

import hgvs
import hgvs.edit
import hgvs.location
import hgvs.sequencevariant
from hgvs.parsers import Parser

# -- Pipeline stages ---------------------------------------------------------

#: Initial parsing of a raw string failed.
STAGE_PARSE = "parse"

#: Formatting the parsed variant failed.
STAGE_FORMAT = "format"

#: Re-parsing the canonical text failed.
STAGE_REPARSE = "reparse"

# -- Difference categories ---------------------------------------------------

#: Object model and text are unchanged.
CATEGORY_NONE = "none"

#: The raw string could not be parsed.
CATEGORY_PARSE_ERROR = "parse_error"

#: The parsed variant could not be formatted.
CATEGORY_FORMAT_ERROR = "format_error"

#: The canonical text could not be re-parsed.
CATEGORY_REPARSE_ERROR = "reparse_error"

#: The canonical text omits the reference sequence accession, which the
#: grammar requires for re-parsing (e.g. an object formatted with ``ac=None``).
CATEGORY_REFERENCE_ACCESSION_OMITTED = "reference_accession_omitted"

#: Variant kind (``c``, ``g``, ``p``, ...) changed.
CATEGORY_VARIANT_TYPE = "variant_type_mismatch"

#: Reference accession changed.
CATEGORY_ACCESSION = "accession_mismatch"

#: Gene symbol changed.
CATEGORY_GENE_SYMBOL = "gene_symbol_changed"

#: Structured edit class/type changed (sub vs delins, etc.).
CATEGORY_EDIT_TYPE = "edit_type_mismatch"

#: Edit reference sequence was present originally but is absent from the
#: canonical representation (typically ``max_ref_length`` display truncation).
CATEGORY_EDIT_REFERENCE_OMITTED = "edit_reference_omitted"

#: Edit reference sequence is present on both sides but differs.
CATEGORY_EDIT_REFERENCE_CHANGED = "edit_reference_changed"

#: Edit alternate sequence differs.
CATEGORY_EDIT_ALTERNATIVE = "edit_alternative_changed"

#: Repeat/frameshift/extension count or length differs.
CATEGORY_EDIT_LENGTH = "edit_length_changed"

#: Copy number differs.
CATEGORY_COPY_NUMBER = "copy_number_changed"

#: Position representation class changed (simple vs intronic vs protein).
CATEGORY_POSITION_TYPE = "position_type_mismatch"

#: Position base coordinate changed.
CATEGORY_POSITION_COORDINATE = "position_coordinate_changed"

#: Intronic offset (``+n``/``-n``) changed.
CATEGORY_INTRON_OFFSET = "intron_offset_changed"

#: Position datum changed (CDS start vs CDS end / ``*`` coordinates).
CATEGORY_POSITION_DATUM = "position_datum_changed"

#: Amino acid at a protein position changed.
CATEGORY_AMINO_ACID = "amino_acid_changed"

#: Uncertainty flag/parenthesization changed.
CATEGORY_UNCERTAINTY = "uncertainty_changed"

#: Conversion (``con``) target differs.
CATEGORY_CONVERSION_TARGET = "conversion_target_changed"

#: Protein extension terminator differs.
CATEGORY_PROTEIN_TERMINATOR = "protein_terminator_changed"

#: Initial-methionine annotation changed.
CATEGORY_INITIAL_METHIONINE = "initial_methionine_changed"

#: Unclassified structural mismatch.
CATEGORY_STRUCTURE = "structure_mismatch"

# -- Cosmetic categories (semantically equivalent, text differs) -------------

#: One- vs three-letter amino acid notation (``R`` vs ``Arg``).
CATEGORY_AMINO_ACID_LETTER_CODE = "amino_acid_letter_code"

#: Terminator notation (``Ter`` vs ``*``).
CATEGORY_TERMINATOR_SYMBOL = "terminator_symbol"

#: Initial-methionine display (``Met1?`` vs ``?``).
CATEGORY_INITIAL_METHIONINE_DISPLAY = "initial_methionine_display"

#: Text differs for no modeled display reason (e.g. tolerated whitespace).
CATEGORY_OTHER_REPRESENTATION = "other_representation"

_INPUT_STRING = "string"
_INPUT_VARIANT = "variant"

#: Priority for choosing a single primary category from field differences.
_SEMANTIC_PRIORITY = [
    CATEGORY_VARIANT_TYPE,
    CATEGORY_ACCESSION,
    CATEGORY_GENE_SYMBOL,
    CATEGORY_EDIT_TYPE,
    CATEGORY_EDIT_REFERENCE_OMITTED,
    CATEGORY_EDIT_REFERENCE_CHANGED,
    CATEGORY_POSITION_TYPE,
    CATEGORY_POSITION_COORDINATE,
    CATEGORY_INTRON_OFFSET,
    CATEGORY_POSITION_DATUM,
    CATEGORY_AMINO_ACID,
    CATEGORY_EDIT_ALTERNATIVE,
    CATEGORY_EDIT_LENGTH,
    CATEGORY_COPY_NUMBER,
    CATEGORY_PROTEIN_TERMINATOR,
    CATEGORY_INITIAL_METHIONINE,
    CATEGORY_UNCERTAINTY,
    CATEGORY_CONVERSION_TARGET,
    CATEGORY_STRUCTURE,
]

_COSMETIC_PRIORITY = [
    CATEGORY_AMINO_ACID_LETTER_CODE,
    CATEGORY_TERMINATOR_SYMBOL,
    CATEGORY_INITIAL_METHIONINE_DISPLAY,
    CATEGORY_OTHER_REPRESENTATION,
]

_POSITION_TYPES = (
    hgvs.location.SimplePosition,
    hgvs.location.BaseOffsetPosition,
    hgvs.location.AAPosition,
    hgvs.location.Interval,
    hgvs.location.BaseOffsetInterval,
)

_FIELD_CATEGORY = {
    "ac": CATEGORY_ACCESSION,
    "gene": CATEGORY_GENE_SYMBOL,
    "type": CATEGORY_VARIANT_TYPE,
    "uncertain": CATEGORY_UNCERTAINTY,
    "base": CATEGORY_POSITION_COORDINATE,
    "offset": CATEGORY_INTRON_OFFSET,
    "datum": CATEGORY_POSITION_DATUM,
    "aa": CATEGORY_AMINO_ACID,
    "alt": CATEGORY_EDIT_ALTERNATIVE,
    "min": CATEGORY_EDIT_LENGTH,
    "max": CATEGORY_EDIT_LENGTH,
    "length": CATEGORY_EDIT_LENGTH,
    "copy": CATEGORY_COPY_NUMBER,
    "aaterm": CATEGORY_PROTEIN_TERMINATOR,
    "init_met": CATEGORY_INITIAL_METHIONINE,
    "from_ac": CATEGORY_CONVERSION_TARGET,
    "from_type": CATEGORY_CONVERSION_TARGET,
    "from_pos": CATEGORY_CONVERSION_TARGET,
}


@attr.s(slots=True, frozen=True)
class RoundTripAudit:
    """Result of a :func:`audit_roundtrip` run.

    All attributes are primitive or tuples of primitive dicts, so
    :meth:`to_dict` yields JSON-serializable data even when parsing or
    re-parsing fails.
    """

    #: ``"string"`` or ``"variant"`` -- how the audited input was supplied.
    input_kind = attr.ib()

    #: The original HGVS text. For variant input this is the object rendered
    #: with global settings (no *conf*); ``None`` if even that failed.
    original_text = attr.ib(default=None)

    #: Text produced by ``SequenceVariant.format(conf)``; ``None`` if the
    #: pipeline failed before or during formatting.
    canonical_text = attr.ib(default=None)

    #: Variant type (``c``, ``g``, ``p``, ...) when parsing succeeded.
    variant_type = attr.ib(default=None)

    #: Whether original and re-parsed canonical variants are semantically
    #: equal by object type and field semantics.
    semantic_equivalent = attr.ib(default=False)

    #: Primary difference category (``"none"`` when fully identical).
    difference_category = attr.ib(default=CATEGORY_NONE)

    #: All difference categories in priority order, including cosmetic ones.
    difference_categories = attr.ib(default=())

    #: Structured field-level differences: dicts with ``path``, ``kind``,
    #: ``original`` and ``canonical`` keys.
    differences = attr.ib(default=())

    #: First failing pipeline stage: ``"parse"``, ``"format"``,
    #: ``"reparse"`` or ``None``.
    error_stage = attr.ib(default=None)

    #: Exception type name at the failing stage, if any.
    error_type = attr.ib(default=None)

    #: Exception message at the failing stage, if any.
    error_message = attr.ib(default=None)

    def to_dict(self):
        """Return a JSON-serializable dict of this report."""
        return {
            "input_kind": self.input_kind,
            "original_text": self.original_text,
            "canonical_text": self.canonical_text,
            "variant_type": self.variant_type,
            "semantic_equivalent": self.semantic_equivalent,
            "difference_category": self.difference_category,
            "difference_categories": list(self.difference_categories),
            "differences": [dict(d) for d in self.differences],
            "error_stage": self.error_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def audit_roundtrip(variant, conf=None, *, parser=None):
    """Audit the parse -> format -> re-parse round trip of one variant.

    :param variant: an HGVS string or a
        :class:`hgvs.sequencevariant.SequenceVariant` instance
    :param conf: optional dict of formatting options forwarded to
        ``SequenceVariant.format`` (e.g. ``{"max_ref_length": None,
        "p_3_letter": False}``).  Display only; it does not trigger
        normalization or external reference sequence access.
    :param parser: optional existing :class:`hgvs.parsers.Parser`
        instance; a default parser is created lazily otherwise

    :returns: :class:`RoundTripAudit`

    :raises TypeError: if *variant* is neither a string nor a
        ``SequenceVariant``
    """
    if isinstance(variant, hgvs.sequencevariant.SequenceVariant):
        return _audit_variant(variant, conf, parser)
    if isinstance(variant, str):
        return _audit_string(variant, conf, parser)
    msg = (
        "variant must be an HGVS string or a SequenceVariant instance, "
        f"not {type(variant).__name__}"
    )
    raise TypeError(msg)


def semantic_differences(original, canonical):
    """Compare two variants by object type and field semantics.

    Differences are derived from the structured object graph (accession,
    coordinates with bases/offsets/datums, edit type/reference/alternate,
    uncertainty flags, ...) -- never from a textual diff.

    :returns: list of dicts with ``path`` (dotted field path), ``kind``
        (a ``CATEGORY_*`` value), ``original`` and ``canonical``
        (serializable snapshots of the differing values)
    """
    return list(_diff(original, canonical, "variant"))


# -- Pipeline implementation -------------------------------------------------


def _audit_string(raw, conf, parser):
    parser = _resolve_parser(parser)
    try:
        parsed = parser.parse(raw)
    except Exception as exc:  # report must be returned even for bad input
        return _error_report(
            input_kind=_INPUT_STRING,
            original_text=raw,
            stage=STAGE_PARSE,
            exc=exc,
            category=CATEGORY_PARSE_ERROR,
        )
    return _audit_parsed(parsed, conf, parser, _INPUT_STRING, raw)


def _audit_variant(variant_obj, conf, parser):
    # Rendering with global settings is only used to populate
    # original_text; failures there must not abort the audit.
    original_text = _safe_format(variant_obj, None)
    return _audit_parsed(variant_obj, conf, parser, _INPUT_VARIANT, original_text)


def _audit_parsed(variant_obj, conf, parser, input_kind, original_text):
    try:
        canonical_text = variant_obj.format(conf)
    except Exception as exc:  # report must be returned even when formatting fails
        return _error_report(
            input_kind=input_kind,
            original_text=original_text,
            stage=STAGE_FORMAT,
            exc=exc,
            category=CATEGORY_FORMAT_ERROR,
            variant_type=variant_obj.type,
        )

    parser = _resolve_parser(parser)
    try:
        reparsed = parser.parse(canonical_text)
    except Exception as exc:  # report must be returned even when re-parsing fails
        category = (
            CATEGORY_REFERENCE_ACCESSION_OMITTED if not variant_obj.ac else CATEGORY_REPARSE_ERROR
        )
        return _error_report(
            input_kind=input_kind,
            original_text=original_text,
            stage=STAGE_REPARSE,
            exc=exc,
            category=category,
            canonical_text=canonical_text,
            variant_type=variant_obj.type,
        )

    differences = semantic_differences(variant_obj, reparsed)
    if differences:
        categories = _ordered_categories((d["kind"] for d in differences), _SEMANTIC_PRIORITY)
        return RoundTripAudit(
            input_kind=input_kind,
            original_text=original_text,
            canonical_text=canonical_text,
            variant_type=variant_obj.type,
            semantic_equivalent=False,
            difference_category=categories[0],
            difference_categories=tuple(categories),
            differences=tuple(differences),
        )

    categories = _cosmetic_categories(variant_obj, original_text, canonical_text, conf)
    return RoundTripAudit(
        input_kind=input_kind,
        original_text=original_text,
        canonical_text=canonical_text,
        variant_type=variant_obj.type,
        semantic_equivalent=True,
        difference_category=categories[0],
        difference_categories=tuple(categories),
    )


def _error_report(
    *,
    input_kind,
    stage,
    exc,
    category,
    original_text=None,
    canonical_text=None,
    variant_type=None,
):
    return RoundTripAudit(
        input_kind=input_kind,
        original_text=original_text,
        canonical_text=canonical_text,
        variant_type=variant_type,
        semantic_equivalent=False,
        difference_category=category,
        difference_categories=(category,),
        error_stage=stage,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


def _resolve_parser(parser):
    return parser if parser is not None else _get_default_parser()


@lru_cache(maxsize=1)
def _get_default_parser():
    return Parser()


def _safe_format(variant_obj, conf):
    try:
        return variant_obj.format(conf)
    except Exception:  # best-effort display of a caller-supplied object
        return None


# -- Cosmetic attribution ----------------------------------------------------


def _cosmetic_categories(variant_obj, original_text, canonical_text, conf):
    if original_text == canonical_text:
        return [CATEGORY_NONE]

    categories = []
    if variant_obj.type == "p":
        categories.extend(
            _protein_display_dimensions(variant_obj, original_text, canonical_text, conf)
        )
    if not categories:
        categories.append(CATEGORY_OTHER_REPRESENTATION)
    return _ordered_categories(categories, _COSMETIC_PRIORITY)


def _protein_display_dimensions(variant_obj, original_text, canonical_text, conf):
    del conf  # attribution searches every display combination, not only effective ones
    dimensions = (
        ("p_3_letter", CATEGORY_AMINO_ACID_LETTER_CODE),
        ("p_term_asterisk", CATEGORY_TERMINATOR_SYMBOL),
        ("p_init_met", CATEGORY_INITIAL_METHIONINE_DISPLAY),
    )

    found = []
    for key, category in dimensions:
        # A display dimension explains the textual difference when no single
        # value of it can render both texts, allowing the other two
        # dimensions to vary freely.  E.g. "G528*"->"Gly528Ter" is fully
        # explained by the letter-code dimension: at one-letter mode "*" is
        # not a separate terminator choice.
        others = [name for name, _ in dimensions if name != key]
        original_values = _renderable_values(variant_obj, key, others, original_text)
        canonical_values = _renderable_values(variant_obj, key, others, canonical_text)
        if original_values and canonical_values and not (original_values & canonical_values):
            found.append(category)
    return found


def _renderable_values(variant_obj, key, other_keys, target_text):
    """Values of *key* (other display dims varying) that render *target_text*."""
    matches = set()
    for value in (True, False):
        for other_a in (True, False):
            for other_b in (True, False):
                render_conf = {
                    key: value,
                    other_keys[0]: other_a,
                    other_keys[1]: other_b,
                }
                if variant_obj.format(render_conf) == target_text:
                    matches.add(value)
    return matches


# -- Structural semantic comparison ------------------------------------------


def _diff(original, canonical, path):
    if original is None or canonical is None:
        if original is not canonical:
            yield _difference(path, _missing_kind(path), original, canonical)
        return

    if type(original) is not type(canonical):
        yield _difference(path, _type_mismatch_kind(path, original, canonical), original, canonical)
        return

    if attr.has(type(original)):
        yield from _diff_attrs(original, canonical, path)
        return

    if isinstance(original, enum.Enum):
        if original is not canonical:
            yield _difference(path, _scalar_field_kind(path), original, canonical)
        return

    if isinstance(original, (list, tuple)):
        if len(original) != len(canonical):
            yield _difference(
                path, CATEGORY_STRUCTURE, f"len={len(original)}", f"len={len(canonical)}"
            )
            return
        for i, (a, b) in enumerate(zip(original, canonical, strict=False)):
            yield from _diff(a, b, f"{path}[{i}]")
        return

    if original != canonical:
        yield _difference(path, _scalar_field_kind(path), original, canonical)


def _diff_attrs(original, canonical, path):
    skip = set()
    # NARefAlt identity edits fold ref and alt together into "="; when the
    # reference is omitted by display, both fields change for one reason,
    # so report the reference side only.
    if (
        path.endswith(".edit")
        and isinstance(original, hgvs.edit.NARefAlt)
        and original.type == "identity"
        and canonical.type == "identity"
        and original.ref != canonical.ref
    ):
        skip.add("alt")

    for field in attr.fields(type(original)):
        if field.name in skip:
            continue
        a = getattr(original, field.name)
        b = getattr(canonical, field.name)
        field_path = f"{path}.{field.name}"
        if field.name == "ref" and path.endswith(".edit"):
            # Classify edit reference sequences from the field itself:
            # missing on both sides (None and "" are equivalent), dropped
            # by display configuration, or present but changed.
            if _is_empty(a) and _is_empty(b):
                continue
            if _is_empty(a) != _is_empty(b):
                yield _difference(field_path, CATEGORY_EDIT_REFERENCE_OMITTED, a, b)
                continue
            if a != b:
                yield _difference(field_path, CATEGORY_EDIT_REFERENCE_CHANGED, a, b)
                continue
        yield from _diff(a, b, field_path)


def _difference(path, kind, original, canonical):
    return {
        "path": path,
        "kind": kind,
        "original": _snapshot(original),
        "canonical": _snapshot(canonical),
    }


def _is_empty(value):
    return value is None or value == ""


def _snapshot(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, (list, tuple)):
        return [_snapshot(v) for v in value]
    return str(value)


def _missing_kind(path):
    if path.endswith(".edit"):
        return CATEGORY_EDIT_TYPE
    if path.endswith((".pos", ".start", ".end")):
        return CATEGORY_POSITION_TYPE
    field = path.rsplit(".", 1)[-1]
    return _FIELD_CATEGORY.get(field, CATEGORY_STRUCTURE)


def _type_mismatch_kind(path, original, canonical):
    if path.endswith(".edit"):
        return CATEGORY_EDIT_TYPE
    if isinstance(original, _POSITION_TYPES) or isinstance(canonical, _POSITION_TYPES):
        return CATEGORY_POSITION_TYPE
    return CATEGORY_STRUCTURE


def _scalar_field_kind(path):
    if ".edit.from_" in path:
        return CATEGORY_CONVERSION_TARGET
    field = path.rsplit(".", 1)[-1]
    if field == "ref" and ".edit" in path:
        return CATEGORY_EDIT_REFERENCE_CHANGED
    return _FIELD_CATEGORY.get(field, CATEGORY_STRUCTURE)


def _ordered_categories(categories, priority):
    index = {name: i for i, name in enumerate(priority)}
    unique = []
    for name in categories:
        if name not in unique:
            unique.append(name)
    unknown = [name for name in unique if name not in index]
    known = sorted((name for name in unique if name in index), key=index.get)
    return known + unknown


# <LICENSE>
# Copyright 2024 HGVS Contributors (https://github.com/biocommons/hgvs)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# </LICENSE>
