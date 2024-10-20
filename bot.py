#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.audio.vad.silero import SileroVAD
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService, OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport
from openai.types.chat import ChatCompletionToolParam

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


phone_number = "+12012345678"



async def main():
    global task
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Waiting On Hold robot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                camera_out_enabled=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            #
            # voice_id="a0e99841-438c-4a64-b679-ae501e7d6091", # Barbershop man
            # voice_id="36b42fcb-60c5-4bec-b077-cb1a00a92ec6",  # British pilot over intercom, butler
            # voice_id="bf991597-6c13-47e4-8411-91ec2de5c466",  # Scarlett Johansson
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British woman
        )

        messages = [
            {
                "role": "system",
                #
                # English
                #
                # "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. Keep all your response to 12 words or fewer.",
                "content": "You are my assistant. We are making a phone call and waiting on hold. Your Job is to not say anything until we are actually talking to a human. You can tell that we are talking to a human because they will say things like 'hi, my name is [name], how can I help you?' or 'thank you for calling, how can I help you today?' Then call tool haveHuman() to let me know that we are talking to a human."
                # Once we are talking to a human, Tell them that you are grabbing your boss, to hang on one moment, and apologize for the delay. Do your best to keep them on the phone until I take over. Keep all your response to 12 words or fewer.
                #
                # Spanish
                #
                # "content": "Eres Chatbot, un amigable y útil robot. Tu objetivo es demostrar tus capacidades de una manera breve. Tus respuestas se convertiran a audio así que nunca no debes incluir caracteres especiales. Contesta a lo que el usuario pregunte de una manera creativa, útil y breve. Empieza por presentarte a ti mismo.",

            },
        ]

        ##################
        # Function call to hand off to user
        ##################
        async def have_human(function_name, tool_call_id, args, llm, context, result_callback):
            logger.info("have_human on the line, LLM can stop")
            logger.info(f"does task exist? It should! Here it is: {task}")
            return

            # turn off LLM
            task.cancel()

            # start a new pipeline without LLM
            pipeline = Pipeline(
                [
                    transport.input(),
                    vad,
                    context_aggregator.user(),
                    context_aggregator.assistant(),
                ]
            )
            task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))
            logger.info("pipeline created with no LLM")

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "have_human",
                    "description": "Let the user know that we are talking to a human",
                    "parameters": {},
                },
            )
        ]

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        llm.register_function("have_human", have_human)
        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)

        vad = SileroVAD()
        pipeline = Pipeline(
            [
                transport.input(),
                vad,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            logger.info(f"Participant joined: {participant}, starting phone call")

            # init phone call
            transport.start_dialout(
                settings={
                    "phoneNumber": phone_number,
                }
            )

            # uncomment to allow user to talk to the bot later on
            transport.capture_participant_transcription(participant["id"])
            # await task.queue_frames([LLMMessagesFrame(messages)])
            await task.queue_frames([context_aggregator.user().get_context_frame()])


        @transport.event_handler("on_dialout_answered")
        async def on_dialout_answered(transport, participant):
            logger.info(f"dialout answered for {phone_number}")

            # transcribe the phone call
            transport.capture_participant_transcription(participant["participantId"])
            # await task.queue_frames([LLMMessagesFrame(messages)])
            await task.queue_frames([context_aggregator.user().get_context_frame()])



        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(transport, state):
            print(f"Call state updated: {state}")
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
