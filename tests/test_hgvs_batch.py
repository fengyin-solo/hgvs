import os
import unittest

import pytest

import hgvs.dataproviders.uta
import hgvs.exceptions
import hgvs.parsers
import hgvs.sequencevariant
import hgvs.validator
from hgvs.batch import (
    STAGE_EXTRINSIC,
    STAGE_INTRINSIC,
    STAGE_PARSE,
    STAGES,
    STATUS_ERROR,
    STATUS_SUCCESS,
    BatchValidator,
    validate_batch,
)
from hgvs.exceptions import HGVSDataNotAvailableError
from support import CACHE

hdp = hgvs.dataproviders.uta.connect(mode=os.environ.get("HGVS_CACHE_MODE", "run"), cache=CACHE)

VALID_C = "NM_001005405.2:c.6C>A"
VALID_G = "NC_000001.10:g.156104193G>A"
BAD_PARSE = "this is not an hgvs variant"
BAD_INTRINSIC = "NC_000007.13:g.36561664_36561663A>T"
BAD_EXTRINSIC = "NC_000001.10:g.156104193C>A"
# passes both validators only when strict=False (intronic reference cannot be checked)
NONSTRICT_ONLY = "NM_000030.2:c.679_680+2delAAGT"


_OUTAGE_MESSAGE = "simulated external data source outage"


class _FailingDataProvider:
    """Minimal data provider stand-in whose external lookups always fail.

    ExtrinsicValidator only reaches hdp.get_seq/get_tx_identity_info for the
    genomic variants used here, and VariantMapper stores the provider
    without calling it at construction time.
    """

    def _fail(self, *_args, **_kwargs):
        raise HGVSDataNotAvailableError(_OUTAGE_MESSAGE)

    get_seq = _fail
    get_tx_identity_info = _fail


@pytest.mark.validation
class Test_BatchValidator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = hgvs.parsers.Parser()
        cls.bv = BatchValidator(hdp)

    # -- basic reporting ---------------------------------------------------

    def test_empty_batch(self):
        report = self.bv.validate([])
        self.assertEqual(report.total, 0)
        self.assertEqual(report.processed, 0)
        self.assertEqual(report.succeeded, 0)
        self.assertEqual(report.failed, 0)
        self.assertFalse(report.aborted)
        self.assertEqual(report.unprocessed, 0)
        self.assertEqual(list(report), [])
        self.assertEqual(report.failures_by_stage, dict.fromkeys(STAGES, 0))
        self.assertEqual(report.errors_by_type, {})
        self.assertIn("total=0", report.summary())

    def test_mixed_good_and_bad_preserves_order_and_stages(self):
        inputs = [VALID_C, BAD_PARSE, BAD_INTRINSIC, BAD_EXTRINSIC, VALID_G]
        report = self.bv.validate(inputs)

        self.assertEqual(report.total, len(inputs))
        self.assertEqual(report.processed, len(inputs))
        self.assertFalse(report.aborted)
        self.assertEqual([item.index for item in report], [0, 1, 2, 3, 4])
        self.assertEqual(
            [item.status for item in report],
            [
                STATUS_SUCCESS,
                STATUS_ERROR,
                STATUS_ERROR,
                STATUS_ERROR,
                STATUS_SUCCESS,
            ],
        )
        self.assertEqual(
            [item.stage for item in report],
            [None, STAGE_PARSE, STAGE_INTRINSIC, STAGE_EXTRINSIC, None],
        )
        self.assertEqual(report.succeeded, 2)
        self.assertEqual(report.failed, 3)
        self.assertEqual(
            report.failures_by_stage,
            {STAGE_PARSE: 1, STAGE_INTRINSIC: 1, STAGE_EXTRINSIC: 1},
        )
        self.assertEqual(report.errors_by_type["HGVSParseError"], 1)

    def test_success_retains_variant_and_normalized_text(self):
        report = self.bv.validate([VALID_C])
        (item,) = report.successes
        self.assertTrue(item.ok)
        self.assertIsInstance(item.variant, hgvs.sequencevariant.SequenceVariant)
        self.assertEqual(item.normalized, VALID_C)
        self.assertEqual(item.normalized, str(item.variant))
        self.assertIsNone(item.stage)
        self.assertIsNone(item.error_type)
        self.assertIsNone(item.error_message)

        # already-parsed variants are validated as-is and returned by identity
        parsed = self.parser.parse(VALID_G)
        (item_from_obj,) = self.bv.validate([parsed]).successes
        self.assertIs(item_from_obj.variant, parsed)
        self.assertEqual(item_from_obj.normalized, VALID_G)

    def test_duplicate_inputs_are_processed_independently(self):
        report = self.bv.validate([VALID_C, VALID_C])
        self.assertEqual(report.succeeded, 2)
        first, second = report.results
        self.assertEqual(first.index, 0)
        self.assertEqual(second.index, 1)
        self.assertEqual(first.normalized, second.normalized)
        # each occurrence is parsed independently
        self.assertIsNot(first.variant, second.variant)

        dup_bad = self.bv.validate([BAD_PARSE, BAD_PARSE])
        self.assertEqual(dup_bad.failed, 2)
        self.assertEqual([item.stage for item in dup_bad.failures], [STAGE_PARSE, STAGE_PARSE])

    def test_iterable_inputs_are_supported(self):
        report = self.bv.validate(iter([VALID_C, BAD_PARSE]))
        self.assertEqual(report.total, 2)
        self.assertEqual(report.succeeded, 1)
        self.assertEqual(report.failed, 1)

    def test_single_unwrapped_item_is_rejected(self):
        with self.assertRaises(TypeError):
            self.bv.validate(VALID_C)
        with self.assertRaises(TypeError):
            self.bv.validate(self.parser.parse(VALID_C))

    # -- failure classification --------------------------------------------

    def test_parse_failures_are_errors_even_without_strict(self):
        for validator in (self.bv, BatchValidator(hdp, strict=False)):
            report = validator.validate([BAD_PARSE])
            (item,) = report.failures
            self.assertEqual(item.status, STATUS_ERROR)
            self.assertEqual(item.stage, STAGE_PARSE)
            self.assertEqual(item.error_type, "HGVSParseError")
            self.assertTrue(item.error_message)
            self.assertIsNone(item.variant)
            self.assertIsNone(item.normalized)

    def test_type_mismatch_is_a_parse_stage_failure(self):
        for bad_input in (42, None, object(), b"NM_001005405.2:c.6C>A"):
            report = self.bv.validate([bad_input])
            (item,) = report.failures
            self.assertEqual(item.stage, STAGE_PARSE)
            self.assertEqual(item.error_type, "TypeError")
            self.assertIn(type(bad_input).__name__, item.error_message)
            self.assertIn("SequenceVariant", item.error_message)

    def test_intrinsic_failure_is_attributed_to_intrinsic_stage(self):
        report = self.bv.validate([BAD_INTRINSIC])
        (item,) = report.failures
        self.assertEqual(item.stage, STAGE_INTRINSIC)
        self.assertEqual(item.error_type, "HGVSInvalidVariantError")
        self.assertTrue(item.error_message)
        # parsing succeeded, so the variant is retained but no normalized text
        self.assertIsInstance(item.variant, hgvs.sequencevariant.SequenceVariant)
        self.assertIsNone(item.normalized)

    def test_extrinsic_failure_is_attributed_to_extrinsic_stage(self):
        report = self.bv.validate([BAD_EXTRINSIC])
        (item,) = report.failures
        self.assertEqual(item.stage, STAGE_EXTRINSIC)
        self.assertEqual(item.error_type, "HGVSInvalidVariantError")
        self.assertTrue(item.error_message)

    def test_external_data_source_error_is_not_reported_as_syntax_error(self):
        bv = BatchValidator(_FailingDataProvider())
        # string input: parsing succeeds before the data provider is consulted
        report = bv.validate([VALID_G, self.parser.parse(VALID_G)])
        self.assertEqual(report.failed, 2)
        for item in report.failures:
            self.assertEqual(item.stage, STAGE_EXTRINSIC)
            self.assertEqual(item.error_type, "HGVSDataNotAvailableError")
            self.assertNotEqual(item.error_type, "HGVSParseError")
            self.assertIn(_OUTAGE_MESSAGE, item.error_message)
        self.assertEqual(report.failures_by_stage[STAGE_PARSE], 0)

    # -- strict mode --------------------------------------------------------

    def test_strict_mode_is_honored_per_validator(self):
        strict = BatchValidator(hdp, strict=True)
        nonstrict = BatchValidator(hdp, strict=False)

        (strict_item,) = strict.validate([NONSTRICT_ONLY]).failures
        self.assertEqual(strict_item.stage, STAGE_INTRINSIC)

        report = nonstrict.validate([NONSTRICT_ONLY])
        self.assertEqual(report.failed, 0)
        (item,) = report.successes
        # canonical text drops the redundant deleted sequence
        self.assertEqual(item.normalized, str(item.variant))
        self.assertEqual(item.normalized, "NM_000030.2:c.679_680+2del")

        # strict must never downgrade unparseable input to a warning
        parse_in_nonstrict = nonstrict.validate([BAD_PARSE])
        self.assertEqual(parse_in_nonstrict.failed, 1)
        self.assertEqual(parse_in_nonstrict.failures[0].stage, STAGE_PARSE)

    # -- max_errors ---------------------------------------------------------

    def test_max_errors_stops_collecting_at_the_budget(self):
        # budget exhausted by the first input; the rest are never processed
        report = self.bv.validate([BAD_PARSE, VALID_C, VALID_C], max_errors=1)
        self.assertEqual(report.processed, 1)
        self.assertEqual(report.failed, 1)
        self.assertTrue(report.aborted)
        self.assertEqual(report.unprocessed, 2)

        # successes before the budget-breaker are retained
        report = self.bv.validate([VALID_C, BAD_PARSE, VALID_C], max_errors=1)
        self.assertEqual(report.processed, 2)
        self.assertEqual([item.status for item in report], [STATUS_SUCCESS, STATUS_ERROR])
        self.assertTrue(report.aborted)
        self.assertEqual(report.unprocessed, 1)

        # exactly the budget reached on the final input is not an abort
        report = self.bv.validate([BAD_PARSE, BAD_PARSE], max_errors=2)
        self.assertEqual(report.processed, 2)
        self.assertFalse(report.aborted)
        self.assertEqual(report.unprocessed, 0)

        # budget reached before the final input aborts with remainder unprocessed
        report = self.bv.validate([BAD_PARSE, BAD_PARSE, VALID_C], max_errors=2)
        self.assertEqual(report.processed, 2)
        self.assertTrue(report.aborted)
        self.assertEqual(report.unprocessed, 1)

        # a budget larger than the error count does not stop anything
        report = self.bv.validate([BAD_PARSE, VALID_C], max_errors=5)
        self.assertEqual(report.processed, 2)
        self.assertFalse(report.aborted)

    def test_max_errors_zero_processes_nothing(self):
        report = self.bv.validate([VALID_C, BAD_PARSE], max_errors=0)
        self.assertEqual(report.processed, 0)
        self.assertEqual(report.total, 2)
        self.assertTrue(report.aborted)
        self.assertEqual(report.unprocessed, 2)

    def test_max_errors_none_processes_every_input(self):
        report = self.bv.validate([BAD_PARSE, BAD_PARSE, BAD_PARSE], max_errors=None)
        self.assertFalse(report.aborted)
        self.assertEqual(report.processed, 3)
        self.assertEqual(report.failed, 3)

    def test_invalid_max_errors(self):
        with self.assertRaises(ValueError):
            self.bv.validate([VALID_C], max_errors=-1)
        for bad_value in ("1", 1.5, True):
            with self.assertRaises(TypeError):
                self.bv.validate([VALID_C], max_errors=bad_value)

    # -- reporting ----------------------------------------------------------

    def test_summary_is_stable_and_free_of_submitted_input(self):
        inputs = [VALID_C, BAD_EXTRINSIC, BAD_PARSE]
        first = self.bv.validate(inputs).summary()
        second = self.bv.validate(list(inputs)).summary()
        self.assertEqual(first, second)

        report = self.bv.validate(inputs)
        summary = report.summary()
        self.assertIn("total=3", summary)
        self.assertIn("succeeded=1", summary)
        self.assertIn("failed=2", summary)
        self.assertIn("parse=1", summary)
        self.assertIn("extrinsic=1", summary)
        self.assertIn("HGVSInvalidVariantError", summary)
        self.assertIn("HGVSParseError", summary)

        # no raw variant text ...
        self.assertNotIn(VALID_C, summary)
        self.assertNotIn(BAD_EXTRINSIC, summary)
        self.assertNotIn(BAD_PARSE, summary)
        # ... and no exception messages (extrinsic messages embed variant text)
        for item in report.failures:
            self.assertTrue(item.error_message)
            self.assertNotIn(item.error_message, summary)
        # the canonical text is available on the item, just not in the summary
        self.assertEqual(report.successes[0].normalized, VALID_C)

    def test_functional_form_matches_class_form(self):
        inputs = [VALID_C, BAD_PARSE, BAD_EXTRINSIC]
        function_report = validate_batch(inputs, hdp, max_errors=10)
        class_report = BatchValidator(hdp).validate(inputs, max_errors=10)
        self.assertEqual(function_report.summary(), class_report.summary())
        self.assertEqual(
            [(i.index, i.status, i.stage, i.error_type) for i in function_report],
            [(i.index, i.status, i.stage, i.error_type) for i in class_report],
        )

    # -- single-item interfaces remain unchanged ----------------------------

    def test_single_item_interfaces_unchanged(self):
        var = self.parser.parse(VALID_C)
        self.assertEqual(str(var), VALID_C)
        self.assertTrue(hgvs.validator.IntrinsicValidator().validate(var))
        self.assertTrue(hgvs.validator.Validator(hdp).validate(var))
        with self.assertRaises(hgvs.exceptions.HGVSParseError):
            self.parser.parse(BAD_PARSE)
        with self.assertRaises(TypeError):
            hgvs.validator.IntrinsicValidator().validate(VALID_C)


if __name__ == "__main__":
    unittest.main()

# <LICENSE>
# Copyright 2026 HGVS Contributors (https://github.com/biocommons/hgvs)
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
