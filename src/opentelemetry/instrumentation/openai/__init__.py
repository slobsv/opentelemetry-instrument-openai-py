# Copyright The OpenTelemetry Authors
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

"""

Usage
-----

The OpenTelemetry ``openai`` integration traces openai create requests

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.openai import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    import openai
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Write a tagline for an ice cream shop.",
    )

API
---
"""



import logging
import json
from typing import Collection

import openai
from wrapt import wrap_function_wrapper

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap

from opentelemetry.instrumentation.openai.package import _instruments
from opentelemetry.instrumentation.openai.version import __version__
from opentelemetry.trace import SpanKind, get_tracer
from opentelemetry.trace.status import Status, StatusCode


logger = logging.getLogger(__name__)


OBJECTS = [
    "ChatCompletion",
    "Completion",
    "Embedding",
    "Image",
]


def _get_response_attributes(response):
    """flatten response object into otel attributes"""

    attributes = response.copy()
    choices = attributes.pop("choices", None)
    data = attributes.pop("data", None)
    usage = attributes.pop("usage", None)

    if choices:
        attributes.update({
            "prompt_response.messages" : [
                choice.text or choice.message.content for choice in choices
            ],
            "prompt_response.finish_reason": [
                choice.finish_reason for choice in choices
            ]
        })
    if data:
        attributes.update({
            "prompt_response.embedding": [
                datum.embedding for datum in data
            ], 
            "prompt_response.image_url": [
                datum.url for datum in data
            ]
        })
    if usage:
        attributes.update({
            f"usage.{key}": value for key, value in usage.items()
        })

    return attributes



def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, cmd):
        def wrapper(wrapped, instance, args, kwargs):
            # prevent double wrapping
            if hasattr(wrapped, "__wrapped__"):
                return wrapped(*args, **kwargs)

            return func(tracer, cmd, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


@_with_tracer_wrapper
def _wrap_cmd(tracer, cmd, wrapped, instance, args, kwargs):
    with tracer.start_as_current_span(
        cmd, kind=SpanKind.CLIENT, attributes={}
    ) as span:
        try:
            if span.is_recording():
                if kwargs:
                    for key, value in kwargs.items():
                        span.set_attribute(key, value)
        except Exception as ex:  # pylint: disable=broad-except
            logger.warning(
                "Failed to set attributes for openai span %s", str(ex)
            )

        response = wrapped(*args, **kwargs)
        if response:
            for key, value in _get_response_attributes(response).items():
                span.set_attribute(key, value)
            span.set_status(Status(StatusCode.OK))
            
        return response


class OpenAIInstrumentor(BaseInstrumentor):
    """A instrumentor for openai module
    See `BaseInstrumentor`
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Instruments the openai module

        Args:
            **kwargs: Optional arguments
                ``tracer_provider``: a TracerProvider, defaults to global.

        """
        tracer_provider = kwargs.get("tracer_provider")

        tracer = trace.get_tracer(
            __name__, __version__, tracer_provider=tracer_provider
        )

        for obj in OBJECTS:
            wrap_function_wrapper(
                "openai",
                f"{obj}.create",
                _wrap_cmd(tracer, "create")
            )

    def _uninstrument(self, **kwargs):
        for obj in OBJECTS:
            unwrap(
                f"openai.{obj}",
                "create",
            )
