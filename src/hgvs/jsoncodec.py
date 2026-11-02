"""Versioned, structured JSON codec for HGVS variant objects

This module serializes :class:`~hgvs.sequencevariant.SequenceVariant`
objects (and the position, interval, and edit objects they contain) to
plain JSON-compatible data, and reconstructs equivalent objects from
such data.

Unlike re-parsing the formatted HGVS text, this codec preserves the
internal structure that cannot be round-tripped through a string: the
coordinate datum (e.g. CDS start vs. end), intronic offsets,
uncertainty flags, and the concrete edit classes (``NARefAlt``
identity, ``AASub``, ``Repeat``, ``AAExt``, ``Dup``, ``Inv``,
``Conv``, ``NACopy``, and so on).

Every encoded document is an envelope with an explicit schema name,
schema version, and the object type of the payload:

    {
      "schema": "hgvs-variant-json",
      "schema_version": 1,
      "object_type": "SequenceVariant",
      "data": { "object_type": "SequenceVariant", ... }
    }

Each nested object is likewise tagged with an ``object_type`` field.
Payloads with an unknown schema version, an unknown object type, or a
malformed structure are rejected with :class:`HGVSJSONError`.  In
strict mode (the default), unexpected fields are also rejected.

The codec is purely structural: it does not normalize, validate
against a reference, or access any external data source, and encoding
never mutates the object being encoded.

Example:

    >>> import hgvs.parsers
    >>> from hgvs import jsoncodec
    >>> hp = hgvs.parsers.Parser()
    >>> v = hp.parse("NM_001166478.1:c.31_32del2")
    >>> v2 = jsoncodec.loads(jsoncodec.dumps(v))
    >>> v2.format({"max_ref_length": None})
    'NM_001166478.1:c.31_32del2'
    >>> type(v2.posedit.edit).__name__
    'NARefAlt'

"""

import json

from hgvs.edit import (
    AAExt,
    AAFs,
    AARefAlt,
    AASub,
    Conv,
    Dup,
    Inv,
    NACopy,
    NARefAlt,
    Repeat,
)
from hgvs.enums import Datum
from hgvs.exceptions import HGVSJSONError
from hgvs.hgvsposition import HGVSPosition
from hgvs.location import (
    AAPosition,
    BaseOffsetInterval,
    BaseOffsetPosition,
    Interval,
    SimplePosition,
)
from hgvs.posedit import PosEdit
from hgvs.sequencevariant import SequenceVariant

#: Identifies the JSON dialect produced by this module.
SCHEMA_NAME = "hgvs-variant-json"

#: Version of the serialized structure.  Bump on incompatible changes.
SCHEMA_VERSION = 1

_OBJECT_TYPE = "object_type"
_ENVELOPE_KEYS = frozenset({"schema", "schema_version", "object_type", "data"})

_DATUMS = {d.name: d for d in Datum}


############################################################################
# Public API
############################################################################


def dumps(obj, *, indent=None):
    """Serialize a supported HGVS object to a versioned JSON string

    :param obj: a :class:`SequenceVariant`, :class:`HGVSPosition`,
        :class:`PosEdit`, interval, position, or edit object
    :param indent: optional indentation passed through to
        :func:`json.dumps`

    :raises HGVSJSONError: if the object contains unsupported types
    """

    return json.dumps(to_document(obj), indent=indent, ensure_ascii=False)


def loads(data, *, strict=True):
    """Deserialize a versioned JSON string (or parsed document)

    :param data: a JSON string/bytes produced by :func:`dumps`, or an
        already-parsed envelope dict
    :param bool strict: reject unexpected fields when True (the default)

    :raises HGVSJSONError: if the payload is invalid JSON, uses an
        unsupported schema version, contains an unknown object type or
        structure, or (in strict mode) unexpected fields
    """

    if isinstance(data, (str, bytes, bytearray)):
        try:
            document = json.loads(data)
        except json.JSONDecodeError as err:
            msg = f"payload is not valid JSON: {err}"
            raise HGVSJSONError(msg) from err
    elif isinstance(data, dict):
        document = data
    else:
        msg = f"expected a JSON string or object, got {type(data).__name__}"
        raise HGVSJSONError(msg)
    return from_document(document, strict=strict)


def to_document(obj):
    """Encode a supported HGVS object as a versioned envelope dict

    :raises HGVSJSONError: if the object contains unsupported types
    """

    node = _encode_node(obj)
    if node is None:
        msg = "cannot encode None as a top-level HGVS JSON document"
        raise HGVSJSONError(msg)
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "object_type": node[_OBJECT_TYPE],
        "data": node,
    }


def from_document(document, *, strict=True):
    """Decode a versioned envelope dict into an HGVS object

    :raises HGVSJSONError: for unknown schema versions, unknown object
        types, malformed structure, or (in strict mode) extra fields
    """

    if not isinstance(document, dict):
        msg = f"top-level payload must be a JSON object, got {type(document).__name__}"
        raise HGVSJSONError(msg)

    if strict:
        extra = set(document) - _ENVELOPE_KEYS
        if extra:
            msg = f"unexpected top-level field(s): {sorted(extra)}"
            raise HGVSJSONError(msg)

    for key in _ENVELOPE_KEYS:
        if key not in document:
            msg = f"missing required top-level field {key!r}"
            raise HGVSJSONError(msg)

    if document["schema"] != SCHEMA_NAME:
        msg = f"unknown schema {document['schema']!r}; expected {SCHEMA_NAME!r}"
        raise HGVSJSONError(msg)

    version = document["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != SCHEMA_VERSION:
        msg = f"unsupported schema version {version!r}; supported version(s): {SCHEMA_VERSION}"
        raise HGVSJSONError(msg)

    object_type = document["object_type"]
    if not isinstance(object_type, str) or object_type not in _DECODERS:
        if isinstance(object_type, str):
            msg = f"unknown object_type {object_type!r}"
        else:
            msg = f"object_type must be a string, got {type(object_type).__name__}"
        raise HGVSJSONError(msg)

    data = document["data"]
    if not isinstance(data, dict):
        msg = f"envelope 'data' must be a JSON object, got {type(data).__name__}"
        raise HGVSJSONError(msg)
    if data.get(_OBJECT_TYPE) != object_type:
        msg = (
            f"envelope object_type {object_type!r} does not match data object_type "
            f"{data.get(_OBJECT_TYPE)!r}"
        )
        raise HGVSJSONError(msg)

    return _decode_node(data, strict, where="data")


############################################################################
# Encoding
############################################################################


def _tagged(tag, **fields):
    node = {_OBJECT_TYPE: tag}
    node.update(fields)
    return node


def _encode_node(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        # PosEdit permits a bare "0" or "=" edit (protein no-change /
        # unknown protein effect forms produced by the grammar).
        return _tagged("LiteralEdit", value=obj)
    try:
        encoder = _ENCODERS[type(obj)]
    except KeyError:
        msg = f"objects of type {type(obj).__name__} are not supported by the HGVS JSON codec"
        raise HGVSJSONError(msg) from None
    return encoder(obj)


def _encode_simple_position(o):
    return _tagged("SimplePosition", base=o.base, uncertain=o.uncertain)


def _encode_base_offset_position(o):
    return _tagged(
        "BaseOffsetPosition",
        base=o.base,
        offset=o.offset,
        datum=o.datum.name,
        uncertain=o.uncertain,
    )


def _encode_aa_position(o):
    return _tagged("AAPosition", base=o.base, aa=o.aa, uncertain=o.uncertain)


def _encode_interval(o):
    tag = "BaseOffsetInterval" if type(o) is BaseOffsetInterval else "Interval"
    return _tagged(
        tag,
        start=_encode_node(o.start),
        end=_encode_node(o.end),
        uncertain=o.uncertain,
    )


def _encode_narefalt(o):
    return _tagged("NARefAlt", ref=o.ref, alt=o.alt, uncertain=o.uncertain)


def _encode_aarefalt(o):
    return _tagged(
        "AARefAlt",
        ref=o.ref,
        alt=o.alt,
        uncertain=o.uncertain,
        init_met=o.init_met,
    )


def _encode_aasub(o):
    return _tagged(
        "AASub",
        ref=o.ref,
        alt=o.alt,
        uncertain=o.uncertain,
        init_met=o.init_met,
    )


def _encode_aafs(o):
    return _tagged(
        "AAFs",
        ref=o.ref,
        alt=o.alt,
        length=o.length,
        uncertain=o.uncertain,
    )


def _encode_aaext(o):
    return _tagged(
        "AAExt",
        ref=o.ref,
        alt=o.alt,
        aaterm=o.aaterm,
        length=o.length,
        uncertain=o.uncertain,
    )


def _encode_dup(o):
    return _tagged("Dup", ref=o.ref, uncertain=o.uncertain)


def _encode_repeat(o):
    return _tagged(
        "Repeat",
        ref=o.ref,
        min=o.min,
        max=o.max,
        uncertain=o.uncertain,
    )


def _encode_nacopy(o):
    return _tagged("NACopy", copy=o.copy, uncertain=o.uncertain)


def _encode_inv(o):
    return _tagged("Inv", ref=o.ref, uncertain=o.uncertain)


def _encode_conv(o):
    return _tagged(
        "Conv",
        from_ac=o.from_ac,
        from_type=o.from_type,
        from_pos=_encode_node(o.from_pos),
        uncertain=o.uncertain,
    )


def _encode_posedit(o):
    return _tagged(
        "PosEdit",
        pos=_encode_node(o.pos),
        edit=_encode_node(o.edit),
        uncertain=o.uncertain,
    )


def _encode_sequence_variant(o):
    return _tagged(
        "SequenceVariant",
        ac=o.ac,
        type=o.type,
        posedit=_encode_node(o.posedit),
        gene=o.gene,
    )


def _encode_hgvs_position(o):
    return _tagged(
        "HGVSPosition",
        ac=o.ac,
        type=o.type,
        pos=_encode_node(o.pos),
        gene=o.gene,
    )


_ENCODERS = {
    SimplePosition: _encode_simple_position,
    BaseOffsetPosition: _encode_base_offset_position,
    AAPosition: _encode_aa_position,
    Interval: _encode_interval,
    BaseOffsetInterval: _encode_interval,
    NARefAlt: _encode_narefalt,
    AARefAlt: _encode_aarefalt,
    AASub: _encode_aasub,
    AAFs: _encode_aafs,
    AAExt: _encode_aaext,
    Dup: _encode_dup,
    Repeat: _encode_repeat,
    NACopy: _encode_nacopy,
    Inv: _encode_inv,
    Conv: _encode_conv,
    PosEdit: _encode_posedit,
    SequenceVariant: _encode_sequence_variant,
    HGVSPosition: _encode_hgvs_position,
}


############################################################################
# Decoding
############################################################################


def _prepare(node, tag, fields, strict, where):
    """Validate a tagged node and return its field set check results"""

    if not isinstance(node, dict):
        msg = f"{where}: expected a JSON object for {tag!r}, got {type(node).__name__}"
        raise HGVSJSONError(msg)
    actual_type = node.get(_OBJECT_TYPE)
    if actual_type != tag:
        msg = f"{where}: expected object_type {tag!r}, got {actual_type!r}"
        raise HGVSJSONError(msg)
    keys = set(node) - {_OBJECT_TYPE}
    missing = fields - keys
    if missing:
        msg = f"{where} ({tag}): missing required field(s) {sorted(missing)}"
        raise HGVSJSONError(msg)
    if strict:
        extra = keys - fields
        if extra:
            msg = f"{where} ({tag}): unexpected field(s) {sorted(extra)}"
            raise HGVSJSONError(msg)


def _str(value, where, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        msg = f"{where}: expected a string, got {type(value).__name__}"
        raise HGVSJSONError(msg)
    return value


def _int(value, where, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{where}: expected an integer, got {type(value).__name__}"
        raise HGVSJSONError(msg)
    return value


def _bool(value, where):
    if not isinstance(value, bool):
        msg = f"{where}: expected a boolean, got {type(value).__name__}"
        raise HGVSJSONError(msg)
    return value


def _na_ref(value, where):
    """NARefAlt.ref may be a sequence string, a count (string digits or
    integer), or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        msg = f"{where}: expected a string, integer, or null, got {type(value).__name__}"
        raise HGVSJSONError(msg)
    return value


def _decode_node(node, strict, where):
    if node is None:
        return None
    if not isinstance(node, dict):
        msg = f"{where}: expected a JSON object, got {type(node).__name__}"
        raise HGVSJSONError(msg)
    tag = node.get(_OBJECT_TYPE)
    if not isinstance(tag, str):
        msg = f"{where}: missing or non-string object_type"
        raise HGVSJSONError(msg)
    try:
        decoder = _DECODERS[tag]
    except KeyError:
        msg = f"{where}: unknown object_type {tag!r}"
        raise HGVSJSONError(msg) from None
    return decoder(node, strict, where)


def _decode_simple_position(d, strict, where):
    fields = frozenset({"base", "uncertain"})
    _prepare(d, "SimplePosition", fields, strict, where)
    return SimplePosition(
        base=_int(d["base"], f"{where}.base", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_base_offset_position(d, strict, where):
    fields = frozenset({"base", "offset", "datum", "uncertain"})
    _prepare(d, "BaseOffsetPosition", fields, strict, where)
    datum_name = _str(d["datum"], f"{where}.datum")
    if datum_name not in _DATUMS:
        msg = f"{where}.datum: unknown datum {datum_name!r}; expected one of {sorted(_DATUMS)}"
        raise HGVSJSONError(msg)
    return BaseOffsetPosition(
        base=_int(d["base"], f"{where}.base", allow_none=True),
        offset=_int(d["offset"], f"{where}.offset", allow_none=True),
        datum=_DATUMS[datum_name],
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_aa_position(d, strict, where):
    fields = frozenset({"base", "aa", "uncertain"})
    _prepare(d, "AAPosition", fields, strict, where)
    return AAPosition(
        base=_int(d["base"], f"{where}.base", allow_none=True),
        aa=_str(d["aa"], f"{where}.aa", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_interval(d, strict, where):
    tag = "BaseOffsetInterval" if d.get(_OBJECT_TYPE) == "BaseOffsetInterval" else "Interval"
    fields = frozenset({"start", "end", "uncertain"})
    _prepare(d, tag, fields, strict, where)
    cls = BaseOffsetInterval if tag == "BaseOffsetInterval" else Interval
    return cls(
        start=_decode_node(d["start"], strict, f"{where}.start"),
        end=_decode_node(d["end"], strict, f"{where}.end"),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_narefalt(d, strict, where):
    fields = frozenset({"ref", "alt", "uncertain"})
    _prepare(d, "NARefAlt", fields, strict, where)
    return NARefAlt(
        ref=_na_ref(d["ref"], f"{where}.ref"),
        alt=_str(d["alt"], f"{where}.alt", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_aarefalt(d, strict, where):
    fields = frozenset({"ref", "alt", "uncertain", "init_met"})
    _prepare(d, "AARefAlt", fields, strict, where)
    return AARefAlt(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        alt=_str(d["alt"], f"{where}.alt", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
        init_met=_bool(d["init_met"], f"{where}.init_met"),
    )


def _decode_aasub(d, strict, where):
    fields = frozenset({"ref", "alt", "uncertain", "init_met"})
    _prepare(d, "AASub", fields, strict, where)
    return AASub(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        alt=_str(d["alt"], f"{where}.alt", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
        init_met=_bool(d["init_met"], f"{where}.init_met"),
    )


def _decode_aafs(d, strict, where):
    fields = frozenset({"ref", "alt", "length", "uncertain"})
    _prepare(d, "AAFs", fields, strict, where)
    return AAFs(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        alt=_str(d["alt"], f"{where}.alt", allow_none=True),
        length=_int(d["length"], f"{where}.length", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_aaext(d, strict, where):
    fields = frozenset({"ref", "alt", "aaterm", "length", "uncertain"})
    _prepare(d, "AAExt", fields, strict, where)
    return AAExt(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        alt=_str(d["alt"], f"{where}.alt", allow_none=True),
        aaterm=_str(d["aaterm"], f"{where}.aaterm", allow_none=True),
        length=_int(d["length"], f"{where}.length", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_dup(d, strict, where):
    fields = frozenset({"ref", "uncertain"})
    _prepare(d, "Dup", fields, strict, where)
    return Dup(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_repeat(d, strict, where):
    fields = frozenset({"ref", "min", "max", "uncertain"})
    _prepare(d, "Repeat", fields, strict, where)
    return Repeat(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        min=_int(d["min"], f"{where}.min", allow_none=True),
        max=_int(d["max"], f"{where}.max", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_nacopy(d, strict, where):
    fields = frozenset({"copy", "uncertain"})
    _prepare(d, "NACopy", fields, strict, where)
    return NACopy(
        copy=_int(d["copy"], f"{where}.copy", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_inv(d, strict, where):
    fields = frozenset({"ref", "uncertain"})
    _prepare(d, "Inv", fields, strict, where)
    return Inv(
        ref=_str(d["ref"], f"{where}.ref", allow_none=True),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_conv(d, strict, where):
    fields = frozenset({"from_ac", "from_type", "from_pos", "uncertain"})
    _prepare(d, "Conv", fields, strict, where)
    return Conv(
        from_ac=_str(d["from_ac"], f"{where}.from_ac", allow_none=True),
        from_type=_str(d["from_type"], f"{where}.from_type", allow_none=True),
        from_pos=_decode_node(d["from_pos"], strict, f"{where}.from_pos"),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_literal_edit(d, strict, where):
    fields = frozenset({"value"})
    _prepare(d, "LiteralEdit", fields, strict, where)
    return _str(d["value"], f"{where}.value")


def _decode_posedit(d, strict, where):
    fields = frozenset({"pos", "edit", "uncertain"})
    _prepare(d, "PosEdit", fields, strict, where)
    return PosEdit(
        pos=_decode_node(d["pos"], strict, f"{where}.pos"),
        edit=_decode_node(d["edit"], strict, f"{where}.edit"),
        uncertain=_bool(d["uncertain"], f"{where}.uncertain"),
    )


def _decode_sequence_variant(d, strict, where):
    fields = frozenset({"ac", "type", "posedit", "gene"})
    _prepare(d, "SequenceVariant", fields, strict, where)
    return SequenceVariant(
        ac=_str(d["ac"], f"{where}.ac", allow_none=True),
        type=_str(d["type"], f"{where}.type", allow_none=True),
        posedit=_decode_node(d["posedit"], strict, f"{where}.posedit"),
        gene=_str(d["gene"], f"{where}.gene", allow_none=True),
    )


def _decode_hgvs_position(d, strict, where):
    fields = frozenset({"ac", "type", "pos", "gene"})
    _prepare(d, "HGVSPosition", fields, strict, where)
    pos = d["pos"]
    if pos is None:
        msg = f"{where} (HGVSPosition): 'pos' may not be null"
        raise HGVSJSONError(msg)
    return HGVSPosition(
        ac=_str(d["ac"], f"{where}.ac"),
        type=_str(d["type"], f"{where}.type"),
        pos=_decode_node(pos, strict, f"{where}.pos"),
        gene=_str(d["gene"], f"{where}.gene", allow_none=True),
    )


_DECODERS = {
    "SimplePosition": _decode_simple_position,
    "BaseOffsetPosition": _decode_base_offset_position,
    "AAPosition": _decode_aa_position,
    "Interval": _decode_interval,
    "BaseOffsetInterval": _decode_interval,
    "NARefAlt": _decode_narefalt,
    "AARefAlt": _decode_aarefalt,
    "AASub": _decode_aasub,
    "AAFs": _decode_aafs,
    "AAExt": _decode_aaext,
    "Dup": _decode_dup,
    "Repeat": _decode_repeat,
    "NACopy": _decode_nacopy,
    "Inv": _decode_inv,
    "Conv": _decode_conv,
    "LiteralEdit": _decode_literal_edit,
    "PosEdit": _decode_posedit,
    "SequenceVariant": _decode_sequence_variant,
    "HGVSPosition": _decode_hgvs_position,
}


# <LICENSE>
# Copyright 2023 HGVS Contributors (https://github.com/biocommons/hgvs)
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
