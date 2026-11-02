"""batch parsing and validation of HGVS variants

This module provides a batch-oriented front end to the existing single-item
:class:`~hgvs.parsers.Parser`, :class:`~hgvs.validator.IntrinsicValidator`,
and :class:`~hgvs.validator.ExtrinsicValidator` components.  Each input is
processed independently, so one malformed string or unavailable external
reference never aborts processing of the remaining inputs.

Inputs may be HGVS variant strings or already-parsed
:class:`~hgvs.sequencevariant.SequenceVariant` objects (or a mix of both).
Every input produces one :class:`BatchItemResult` in the original input
order; successful items retain the parsed variant and its canonical
(normalized) HGVS text, and failed items record the processing stage, the
exception type, and a human-readable message.

The three failure stages are, in processing order:

* ``"parse"`` -- the input could not be turned into a SequenceVariant
  (including inputs that are neither strings nor SequenceVariants).  Parse
  failures are always errors, even when ``strict=False``; they are never
  downgraded to warnings.
* ``"intrinsic"`` -- internal consistency checks performed by
  :class:`~hgvs.validator.IntrinsicValidator`.
* ``"extrinsic"`` -- validation against external reference data performed
  by :class:`~hgvs.validator.ExtrinsicValidator`.  External data source
  failures (e.g. :class:`~hgvs.exceptions.HGVSDataNotAvailableError`) are
  attributed to this stage and are never misreported as syntax errors.

Example::

    import hgvs.dataproviders.uta
    from hgvs.batch import BatchValidator

    hdp = hgvs.dataproviders.uta.connect()
    bv = BatchValidator(hdp)
    report = bv.validate(["NM_001005405.2:c.6C>A", "not an hgvs string"], max_errors=10)
    print(report.summary())
    for item in report:
        if item.ok:
            load(item.normalized)
        else:
            route_to_review(item.index, item.stage, item.error_type, item.error_message)

"""

from collections import Counter
from collections.abc import Iterable

import attr

import hgvs
import hgvs.sequencevariant
from hgvs.parsers import Parser
from hgvs.validator import ExtrinsicValidator, IntrinsicValidator

#: Input could not be parsed into a SequenceVariant.
STAGE_PARSE = "parse"
#: Parsed variant failed internal consistency validation.
STAGE_INTRINSIC = "intrinsic"
#: Parsed variant failed validation against external reference data.
STAGE_EXTRINSIC = "extrinsic"

#: Processing stages in execution order; used for stable reporting.
STAGES = (STAGE_PARSE, STAGE_INTRINSIC, STAGE_EXTRINSIC)

#: Item passed all requested processing stages.
STATUS_SUCCESS = "success"
#: Item raised an exception at some processing stage.
STATUS_ERROR = "error"


@attr.s(frozen=True, slots=True)
class BatchItemResult:
    """Result of processing a single batch input.

    Successful items have ``status == "success"``, a non-None ``variant``,
    and canonical ``normalized`` text.  Failed items have
    ``status == "error"`` with ``stage``, ``error_type``, and
    ``error_message`` set; ``variant`` is retained when parsing had already
    succeeded (i.e. for intrinsic/extrinsic failures), and is ``None`` for
    parse-stage failures.
    """

    #: Zero-based position of the input in the submitted batch.
    index = attr.ib()
    #: Either :data:`STATUS_SUCCESS` or :data:`STATUS_ERROR`.
    status = attr.ib()
    #: Parsed SequenceVariant, or None when parsing failed.
    variant = attr.ib(default=None)
    #: Canonical HGVS text (``str(variant)``) for successful items.
    normalized = attr.ib(default=None)
    #: Failure stage (see :data:`STAGES`), or None for successful items.
    stage = attr.ib(default=None)
    #: Name of the exception class raised for a failed item.
    error_type = attr.ib(default=None)
    #: Human-readable exception message for a failed item.
    error_message = attr.ib(default=None)

    @property
    def ok(self):
        """True when the item completed every processing stage successfully."""
        return self.status == STATUS_SUCCESS


@attr.s(frozen=True, slots=True)
class BatchValidationResult:
    """Report of a batch validation run.

    Counts and breakdowns are derived from the retained per-item results,
    so they are always internally consistent.  Only inputs actually
    processed have results; when processing stops early because of
    ``max_errors``, :attr:`unprocessed` gives the number of skipped inputs
    and :attr:`aborted` is True.
    """

    #: Per-item results for processed inputs, in original input order.
    results = attr.ib()
    #: Total number of inputs submitted to the batch.
    total = attr.ib()
    #: Error budget that was in effect (None means unlimited).
    max_errors = attr.ib(default=None)

    def __iter__(self):
        return iter(self.results)

    @property
    def processed(self):
        """Number of inputs for which a result was collected."""
        return len(self.results)

    @property
    def unprocessed(self):
        """Number of inputs not reached because processing stopped early."""
        return self.total - len(self.results)

    @property
    def aborted(self):
        """True when one or more inputs were skipped due to max_errors."""
        return self.unprocessed > 0

    @property
    def succeeded(self):
        """Number of successful items."""
        return sum(1 for item in self.results if item.ok)

    @property
    def failed(self):
        """Number of failed items."""
        return sum(1 for item in self.results if not item.ok)

    @property
    def successes(self):
        """Tuple of successful item results, in original input order."""
        return tuple(item for item in self.results if item.ok)

    @property
    def failures(self):
        """Tuple of failed item results, in original input order."""
        return tuple(item for item in self.results if not item.ok)

    @property
    def failures_by_stage(self):
        """Failed item counts keyed by stage, in canonical stage order.

        The mapping always contains all stages (with zero counts when no
        failure occurred at a stage).
        """
        counts = Counter(item.stage for item in self.results if not item.ok)
        return {stage: counts.get(stage, 0) for stage in STAGES}

    @property
    def errors_by_type(self):
        """Failed item counts keyed by exception class name.

        Keys are sorted alphabetically so the mapping is stable across runs
        and batches.  Only exception type names are included; the values
        never contain submitted HGVS text or exception messages.
        """
        counts = Counter(item.error_type for item in self.results if not item.ok)
        return dict(sorted(counts.items()))

    def summary(self):
        """Return a stable, human-readable summary without submitted inputs.

        The summary contains counts only -- total/processed/succeeded/
        failed/unprocessed totals, per-stage failure counts, and
        per-exception-type failure counts.  It never contains raw inputs,
        canonical variant text, or exception messages (which may embed
        variant text), so it is safe for logs and reports that must not
        carry patient-derived data.
        """
        stage_breakdown = " ".join(
            f"{stage}={count}" for stage, count in self.failures_by_stage.items()
        )
        lines = [
            "hgvs batch validation summary: "
            f"total={self.total} processed={self.processed} "
            f"succeeded={self.succeeded} failed={self.failed} "
            f"aborted={str(self.aborted).lower()} unprocessed={self.unprocessed}",
            f"failures by stage: {stage_breakdown}",
        ]
        if self.errors_by_type:
            type_breakdown = " ".join(
                f"{error_type}={count}" for error_type, count in self.errors_by_type.items()
            )
            lines.append(f"failures by type: {type_breakdown}")
        return "\n".join(lines)


class BatchValidator:
    """Parse and validate HGVS variants in bulk, one independent result per input.

    This class composes the same components used by the single-item
    :class:`~hgvs.validator.Validator` -- a :class:`~hgvs.parsers.Parser`,
    :class:`~hgvs.validator.IntrinsicValidator`, and
    :class:`~hgvs.validator.ExtrinsicValidator` -- so single-item and batch
    behavior are identical.  String inputs go through all three stages
    (parse, intrinsic, extrinsic); already-parsed
    :class:`~hgvs.sequencevariant.SequenceVariant` inputs skip parsing.
    """

    def __init__(self, hdp, parser=None, strict=None):
        """Initialize the batch validator.

        :param hdp: HGVS Data Provider Interface-compliant instance
            (see :class:`hgvs.dataproviders.interface.Interface`), used for
            extrinsic validation
        :param parser: optional preconfigured :class:`~hgvs.parsers.Parser`
            (e.g. to select a different grammar); a default ``Parser()`` is
            created when omitted
        :param strict: strict validation flag forwarded to the intrinsic
            and extrinsic validators; None means the package default
            (``hgvs.global_config.validator.strict``)

        """
        self.strict = hgvs.global_config.validator.strict if strict is None else strict
        self._parser = parser if parser is not None else Parser()
        self._intrinsic_validator = IntrinsicValidator(self.strict)
        self._extrinsic_validator = ExtrinsicValidator(hdp, self.strict)

    def validate(self, variants, max_errors=None):
        """Validate an iterable of HGVS strings and/or SequenceVariants.

        Each input is processed in isolation; exceptions are captured as
        failed :class:`BatchItemResult` objects instead of interrupting the
        loop.  Results are returned in the original input order.

        :param variants: iterable of HGVS variant strings or parsed
            SequenceVariants; the iterable is materialized once so that
            generators are supported
        :param max_errors: stop collecting results after this many failed
            items.  None (the default) processes every input.  The item
            that reaches the error budget is included in the results;
            remaining inputs are reported as unprocessed.  ``0`` aborts
            before the first input.  Must be None or a non-negative int.
        :rtype: BatchValidationResult

        """
        if isinstance(variants, (str, hgvs.sequencevariant.SequenceVariant)):
            msg = (
                "variants must be an iterable of HGVS strings or SequenceVariants; "
                "wrap a single item in a list"
            )
            raise TypeError(msg)
        self._validate_max_errors(max_errors)
        items = list(variants)

        results = []
        error_count = 0
        for index, item in enumerate(items):
            if max_errors is not None and error_count >= max_errors:
                break
            result = self._validate_one(index, item)
            results.append(result)
            if not result.ok:
                error_count += 1

        return BatchValidationResult(
            results=tuple(results), total=len(items), max_errors=max_errors
        )

    def _validate_one(self, index, item):
        """Run all applicable stages for one input, capturing exceptions."""
        if isinstance(item, hgvs.sequencevariant.SequenceVariant):
            var = item
        elif isinstance(item, str):
            try:
                var = self._parser.parse(item)
            except Exception as exc:
                # Parse failures are hard errors regardless of strict mode;
                # they must never be downgraded to warnings.
                return _failure(index, STAGE_PARSE, exc)
        else:
            type_error = TypeError(
                f"expected an HGVS variant string or SequenceVariant, got {type(item).__name__}"
            )
            return _failure(index, STAGE_PARSE, type_error)

        for stage, validator in (
            (STAGE_INTRINSIC, self._intrinsic_validator),
            (STAGE_EXTRINSIC, self._extrinsic_validator),
        ):
            try:
                validator.validate(var, self.strict)
            except Exception as exc:
                # External data outages are intentionally not caught at the
                # parse stage: they remain extrinsic failures with their
                # original exception type.
                return _failure(index, stage, exc, var)

        return BatchItemResult(
            index=index,
            status=STATUS_SUCCESS,
            variant=var,
            normalized=str(var),
        )

    @staticmethod
    def _validate_max_errors(max_errors):
        """Validate the caller-supplied error budget."""
        if max_errors is None:
            return
        if isinstance(max_errors, bool) or not isinstance(max_errors, int):
            msg = "max_errors must be None or an integer"
            raise TypeError(msg)
        if max_errors < 0:
            msg = "max_errors must be >= 0"
            raise ValueError(msg)


def _failure(index, stage, exc, variant=None):
    """Build a failed BatchItemResult from a captured exception."""
    return BatchItemResult(
        index=index,
        status=STATUS_ERROR,
        variant=variant,
        stage=stage,
        error_type=type(exc).__name__,
        error_message=str(exc) or type(exc).__name__,
    )


def validate_batch(variants: Iterable, hdp, *, parser=None, strict=None, max_errors=None):
    """Functional form for one-off batch validation.

    This is a convenience wrapper that constructs a
    :class:`BatchValidator` and runs a single batch.  Callers processing
    more than one batch should construct :class:`BatchValidator` directly
    to reuse the underlying parser, validators, and data provider caches.

    :param variants: iterable of HGVS variant strings or parsed
        SequenceVariants
    :param hdp: HGVS Data Provider Interface-compliant instance
    :param parser: optional preconfigured :class:`~hgvs.parsers.Parser`
    :param strict: strict validation flag (None uses the package default)
    :param max_errors: stop after this many failed items (None = unlimited)
    :rtype: BatchValidationResult

    """
    return BatchValidator(hdp, parser=parser, strict=strict).validate(
        variants, max_errors=max_errors
    )


__all__ = [
    "STAGES",
    "STAGE_EXTRINSIC",
    "STAGE_INTRINSIC",
    "STAGE_PARSE",
    "STATUS_ERROR",
    "STATUS_SUCCESS",
    "BatchItemResult",
    "BatchValidationResult",
    "BatchValidator",
    "validate_batch",
]


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
