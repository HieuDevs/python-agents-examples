import logging
import json
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import TypedDict, List, Optional
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RoomOutputOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    noise_cancellation,
    deepgram,
    elevenlabs,
    silero,
    openai,
    tavus,
)

import asyncio


env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
logger = logging.getLogger("avatar")
logger.setLevel(logging.INFO)


class QuizAnswerDict(TypedDict):
    text: str
    is_correct: bool


class QuizQuestionDict(TypedDict):
    text: str
    answers: List[QuizAnswerDict]


@dataclass
class FlashCard:
    """Class to represent a flash card."""

    id: str
    question: str
    answer: str
    is_flipped: bool = False


@dataclass
class QuizAnswer:
    """Class to represent a quiz answer option."""

    id: str
    text: str
    is_correct: bool


@dataclass
class QuizQuestion:
    """Class to represent a quiz question."""

    id: str
    text: str
    answers: List[QuizAnswer]


@dataclass
class Quiz:
    """Class to represent a quiz."""

    id: str
    questions: List[QuizQuestion]


@dataclass
class UserData:
    """Class to store user data during a session."""

    ctx: Optional[JobContext] = None
    flash_cards: List[FlashCard] = field(default_factory=list)
    quizzes: List[Quiz] = field(default_factory=list)

    def reset(self) -> None:
        """Reset session data."""
        # Keep flash cards and quizzes intact

    def add_flash_card(self, question: str, answer: str) -> FlashCard:
        """Add a new flash card to the collection."""
        card = FlashCard(id=str(uuid.uuid4()), question=question, answer=answer)
        self.flash_cards.append(card)
        return card

    def get_flash_card(self, card_id: str) -> Optional[FlashCard]:
        """Get a flash card by ID."""
        for card in self.flash_cards:
            if card.id == card_id:
                return card
        return None

    def flip_flash_card(self, card_id: str) -> Optional[FlashCard]:
        """Flip a flash card by ID."""
        card = self.get_flash_card(card_id)
        if card:
            card.is_flipped = not card.is_flipped
            return card
        return None

    def add_quiz(self, questions: List[QuizQuestionDict]) -> Quiz:
        """Add a new quiz to the collection."""
        quiz_questions = []
        for q in questions:
            answers = []
            for a in q["answers"]:
                answers.append(
                    QuizAnswer(
                        id=str(uuid.uuid4()), text=a["text"], is_correct=a["is_correct"]
                    )
                )
            quiz_questions.append(
                QuizQuestion(id=str(uuid.uuid4()), text=q["text"], answers=answers)
            )

        quiz = Quiz(id=str(uuid.uuid4()), questions=quiz_questions)
        self.quizzes.append(quiz)
        return quiz

    def get_quiz(self, quiz_id: str) -> Optional[Quiz]:
        """Get a quiz by ID."""
        for quiz in self.quizzes:
            if quiz.id == quiz_id:
                return quiz
        return None

    def check_quiz_answers(self, quiz_id: str, user_answers: dict) -> List[tuple]:
        """Check user's quiz answers and return results."""
        quiz = self.get_quiz(quiz_id)
        if not quiz:
            return []

        results = []
        for question in quiz.questions:
            user_answer_id = user_answers.get(question.id)

            # Find the selected answer and the correct answer
            selected_answer = None
            correct_answer = None

            for answer in question.answers:
                if answer.id == user_answer_id:
                    selected_answer = answer
                if answer.is_correct:
                    correct_answer = answer

            is_correct = selected_answer and selected_answer.is_correct
            results.append((question, selected_answer, correct_answer, is_correct))

        return results


class AvatarAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
                You are a friendly English tutor who will first greet and ask for the student's name, then greet them again using their name. After that, you will teach English based on topics chosen by the student.

                Initial greeting sequence:
                • First greeting: "Hello! I'm your English tutor. What's your name?"
                • After getting name: "Nice to meet you, [name]! I'm excited to help you learn English. What topic would you like to practice today?"

                Key responsibilities:
                • Teach vocabulary and phrases based on student's chosen topics
                • Practice conversations relevant to their interests
                • Give clear, simple explanations
                • Focus on natural speaking
                • Create quizzes and flashcards when needed

                Teaching approach:
                • Keep corrections gentle and positive
                • Use clear pronunciation
                • Adapt to student's level
                • Encourage speaking practice
                • Create learning materials based on student's needs

                Response evaluation:
                • After each student response, evaluate their understanding
                • Create flashcards for vocabulary gaps
                • Create quizzes for concept practice
                • Create both for common mistakes
                • Explain why additional materials are provided

                FLASH CARDS:
                Create cards when:
                • Student shows vocabulary gaps in their chosen topic
                • New concepts are introduced
                • Common mistakes are made
                • Cultural context is needed

                Example card creation:
                If student chooses "business English" and struggles with email writing:
                Q: "How to start a business email professionally?"
                A: "I hope this email finds you well."

                QUIZZES:
                Create quizzes when:
                • Student needs practice with topic concepts
                • Multiple related topics need review
                • Common mistakes need addressing
                • Real-world application is needed

                Example quiz creation:
                If student chooses "business meetings" and struggles with greetings:
                ```python
                await self.create_quiz([
                    {
                        "text": "Best greeting for a business meeting?",
                        "answers": [
                            {"text": "Hey, what's up?", "is_correct": False},
                            {"text": "Good morning, it's a pleasure to meet you", "is_correct": True},
                            {"text": "Yo, nice to meet ya", "is_correct": False},
                            {"text": "Hi there, buddy", "is_correct": False}
                        ]
                    }
                ])
                ```

                Conversation flow:
                1. Greet and get student's name
                2. Greet again using their name
                3. Ask about their preferred topic
                4. Teach based on their chosen topic
                5. After each response:
                   - Evaluate understanding
                   - Provide gentle correction if needed
                   - Create appropriate flashcards or quizzes
                   - Explain why you're providing additional materials
                   - Keep the conversation flowing naturally
            """,
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4.1-nano"),
            tts=elevenlabs.TTS(voice_id="21m00Tcm4TlvDq8ikWAM"),
            vad=silero.VAD.load(),
        )

    @function_tool
    async def create_flash_card(
        self, context: RunContext[UserData], question: str, answer: str
    ):
        """Create a new flash card and display it to the user.

        Args:
            question: The question or front side of the flash card
            answer: The answer or back side of the flash card
        """
        userdata = context.userdata
        card = userdata.add_flash_card(question, answer)

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Created a flash card, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Created a flash card, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Created a flash card, but couldn't get the first participant."
        payload = {
            "action": "show",
            "id": card.id,
            "question": card.question,
            "answer": card.answer,
            "index": len(userdata.flash_cards) - 1,
        }

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending flash card payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.flashcard",
            payload=json_payload,
        )

        return f"I've created a flash card with the question: '{question}'"

    @function_tool
    async def flip_flash_card(self, context: RunContext[UserData], card_id: str):
        """Flip a flash card to show the answer or question.

        Args:
            card_id: The ID of the flash card to flip
        """
        userdata = context.userdata
        card = userdata.flip_flash_card(card_id)

        if not card:
            return f"Flash card with ID {card_id} not found."

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Flipped the flash card, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Flipped the flash card, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Flipped the flash card, but couldn't get the first participant."
        payload = {"action": "flip", "id": card.id}

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending flip card payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.flashcard",
            payload=json_payload,
        )

        return f"I've flipped the flash card to show the {'answer' if card.is_flipped else 'question'}"

    @function_tool
    async def create_quiz(
        self, context: RunContext[UserData], questions: List[QuizQuestionDict]
    ):
        """Create a new quiz with multiple choice questions and display it to the user.

        Args:
            questions: A list of question objects. Each question object should have:
                - text: The question text
                - answers: A list of answer objects, each with:
                    - text: The answer text
                    - is_correct: Boolean indicating if this is the correct answer
        """
        userdata = context.userdata
        quiz = userdata.add_quiz(questions)

        # Get the room from the userdata
        if not userdata.ctx or not userdata.ctx.room:
            return f"Created a quiz, but couldn't access the room to send it."

        room = userdata.ctx.room

        # Get the first participant in the room (should be the client)
        participants = room.remote_participants
        if not participants:
            return f"Created a quiz, but no participants found to send it to."

        # Get the first participant from the dictionary of remote participants
        participant = next(iter(participants.values()), None)
        if not participant:
            return f"Created a quiz, but couldn't get the first participant."

        # Format questions for client
        client_questions = []
        for q in quiz.questions:
            client_answers = []
            for a in q.answers:
                client_answers.append({"id": a.id, "text": a.text})
            client_questions.append(
                {"id": q.id, "text": q.text, "answers": client_answers}
            )

        payload = {"action": "show", "id": quiz.id, "questions": client_questions}

        # Make sure payload is properly serialized
        json_payload = json.dumps(payload)
        logger.info(f"Sending quiz payload: {json_payload}")
        await room.local_participant.perform_rpc(
            destination_identity=participant.identity,
            method="client.quiz",
            payload=json_payload,
        )

        return f"I've created a quiz with {len(questions)} questions. Please answer them when you're ready."

    async def on_enter(self):
        await asyncio.sleep(1)
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    # Get metadat from participants
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant name: {participant.name}")
    logger.info(f"Participant identity: {participant.identity}")
    logger.info(f"Participant attributes: {participant.attributes}")
    logger.info(f"Participant metada: {participant.metadata}")
    # logger.info(f"Participant kind: {participant.kind}")
    # logger.info(f"Participant disconnect reason: {participant.disconnect_reason}")
    # logger.info(f"Participant track_publications: {participant.track_publications}")

    agent = AvatarAgent()

    # Create a single AgentSession with userdata
    userdata = UserData(ctx=ctx)
    session = AgentSession[UserData](
        userdata=userdata, turn_detection=MultilingualModel()
    )

    # Create the avatar session
    avatar = tavus.AvatarSession(
        replica_id="ra54d1d861",
        # replica_id="rf4703150052",
        persona_id="p7fb0be3",
    )

    # Register RPC method for flipping flash cards from client
    async def handle_flip_flash_card(rpc_data):
        try:
            logger.info(f"Received flash card flip payload: {rpc_data}")

            # Extract the payload from the RpcInvocationData object
            payload_str = rpc_data.payload
            logger.info(f"Extracted payload string: {payload_str}")

            # Parse the JSON payload
            payload_data = json.loads(payload_str)
            logger.info(f"Parsed payload data: {payload_data}")

            card_id = payload_data.get("id")

            if card_id:
                card = userdata.flip_flash_card(card_id)
                if card:
                    logger.info(
                        f"Flipped flash card {card_id}, is_flipped: {card.is_flipped}"
                    )
                    # Send a message to the user via the agent, we're disabling this for now.
                    # session.generate_reply(user_input=(f"Please describe the {'answer' if card.is_flipped else 'question'}"))
                else:
                    logger.error(f"Card with ID {card_id} not found")
            else:
                logger.error("No card ID found in payload")

            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for payload '{rpc_data.payload}': {e}")
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Error handling flip flash card: {e}")
            return f"error: {str(e)}"

    # Register RPC method for handling quiz submissions
    async def handle_submit_quiz(rpc_data):
        try:
            logger.info(f"Received quiz submission payload: {rpc_data}")

            # Extract the payload from the RpcInvocationData object
            payload_str = rpc_data.payload
            logger.info(f"Extracted quiz submission string: {payload_str}")

            # Parse the JSON payload
            payload_data = json.loads(payload_str)
            logger.info(f"Parsed quiz submission data: {payload_data}")

            quiz_id = payload_data.get("id")
            user_answers = payload_data.get("answers", {})

            if not quiz_id:
                logger.error("No quiz ID found in payload")
                return "error: No quiz ID found in payload"

            # Check the quiz answers
            quiz_results = userdata.check_quiz_answers(quiz_id, user_answers)
            if not quiz_results:
                logger.error(f"Quiz with ID {quiz_id} not found")
                return "error: Quiz not found"

            # Count correct answers
            correct_count = sum(1 for _, _, _, is_correct in quiz_results if is_correct)
            total_count = len(quiz_results)

            # Create a verbal response for the agent to say
            result_summary = (
                f"You got {correct_count} out of {total_count} questions correct."
            )

            # Generate feedback for each question
            feedback_details = []
            for question, selected_answer, correct_answer, is_correct in quiz_results:
                if is_correct:
                    feedback = f"Question: {question.text}\nYour answer: {selected_answer.text} ✓ Correct!"
                else:
                    feedback = f"Question: {question.text}\nYour answer: {selected_answer.text if selected_answer else 'None'} ✗ Incorrect. The correct answer is: {correct_answer.text}"

                    # Create a flash card for incorrectly answered questions
                    card = userdata.add_flash_card(question.text, correct_answer.text)
                    participant = next(
                        iter(ctx.room.remote_participants.values()), None
                    )
                    if participant:
                        flash_payload = {
                            "action": "show",
                            "id": card.id,
                            "question": card.question,
                            "answer": card.answer,
                            "index": len(userdata.flash_cards) - 1,
                        }
                        json_flash_payload = json.dumps(flash_payload)
                        await ctx.room.local_participant.perform_rpc(
                            destination_identity=participant.identity,
                            method="client.flashcard",
                            payload=json_flash_payload,
                        )

                feedback_details.append(feedback)

            detailed_feedback = "\n\n".join(feedback_details)
            full_response = f"{result_summary}\n\n{detailed_feedback}"

            # Have the agent say the results
            session.say(full_response)

            return "success"
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parsing error for quiz submission payload '{rpc_data.payload}': {e}"
            )
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Error handling quiz submission: {e}")
            return f"error: {str(e)}"

    # Register RPC methods - The method names need to match exactly what the client is calling
    logger.info("Registering RPC methods")
    ctx.room.local_participant.register_rpc_method(
        "agent.flipFlashCard", handle_flip_flash_card
    )

    ctx.room.local_participant.register_rpc_method(
        "agent.submitQuiz", handle_submit_quiz
    )

    # Start the avatar with the same session that has userdata
    await avatar.start(session, room=ctx.room)

    # Start the agent session with the same session object
    await session.start(
        room=ctx.room,
        room_output_options=RoomOutputOptions(
            audio_enabled=True,  # Enable audio since we want the avatar to speak
        ),
        room_input_options=RoomInputOptions(
            audio_enabled=True,  # Enable audio since we want the avatar to speak
            noise_cancellation=noise_cancellation.BVC(),
        ),
        agent=agent,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
