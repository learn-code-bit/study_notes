"""
Lecture Notes -> Study Pack   (single file: LangChain + RAG + free LLM)

SETUP (once):
    pip install streamlit langchain-core langchain-text-splitters numpy pydantic pypdf langchain-google-genai
    (Groq instead of Gemini:  pip install langchain-groq        Ollama: pip install langchain-ollama)

RUN:
    1. Paste your free API key into GEMINI_API_KEY below (or type it in the app's sidebar).
    2. streamlit run study_pack_app.py

Do not share or submit this file with a real key inside it.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import socket
import time
import zlib
from typing import Callable, Literal

import numpy as np
import streamlit as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# ======================================================================================
# PASTE YOUR FREE API KEY BETWEEN THE QUOTES (or leave "" and type it in the sidebar)
GEMINI_API_KEY = ""   # free key: https://aistudio.google.com/apikey
GROQ_API_KEY = ""     # free key: https://console.groq.com/keys
# ======================================================================================

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",   # free tier (2.5 models closed to new users); alt: gemini-3.5-flash
    "groq": "openai/gpt-oss-120b",       # free tier (Groq retired llama-3.3-70b in Aug 2026); alt: openai/gpt-oss-20b
    "ollama": "llama3.1:8b",             # local; needs Ollama running
}
KEY_ENV = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}
PIP_HINT = {"gemini": "pip install langchain-google-genai", "groq": "pip install langchain-groq",
            "ollama": "pip install langchain-ollama"}
# Seconds between LLM calls so free-tier requests-per-minute limits are respected.
MIN_INTERVAL = {"gemini": 4.5, "groq": 2.5, "ollama": 0.0}

SAMPLE_NOTES = "Lecture 4: Introduction to Supervised Learning\nProf. Rao - CS 301 Machine Learning\n\n1. What is machine learning?\nMachine learning is a way of building systems that learn patterns from data instead of being explicitly programmed with rules. The three main paradigms are supervised learning, unsupervised learning and reinforcement learning. In supervised learning the training data is labeled, meaning each example comes with the correct answer. In unsupervised learning there are no labels, and the goal is to find structure in the data, for example clustering customers into groups. In reinforcement learning an agent learns by trial and error using rewards.\n\n2. Supervised learning tasks\nThere are two main types of supervised task. Classification predicts a discrete category, such as spam or not spam. Regression predicts a continuous number, such as the price of a house. The model learns a function that maps input features to an output label from the training examples.\n\n3. Train, validation and test sets\nWe never evaluate a model only on the data it was trained on. The data is split into three parts: a training set used to fit the model, a validation set used to tune hyperparameters and choose between models, and a test set used once at the end to estimate performance on unseen data. A common split is 70% training, 15% validation and 15% test. Reusing the test set many times leaks information and gives an over-optimistic estimate, which is called data leakage. When data is limited we use k-fold cross-validation: the data is split into k equal folds, the model is trained on k-1 folds and validated on the remaining fold, and this is repeated k times and the scores are averaged.\n\n4. Overfitting and underfitting\nOverfitting happens when a model memorizes noise in the training data, so training error is low but error on new data is high. Underfitting happens when the model is too simple to capture the pattern, so both training and validation error are high. This is the bias-variance tradeoff: high bias means the model is too simple (underfitting) and high variance means it is too sensitive to the training data (overfitting). Ways to reduce overfitting include getting more data, using a simpler model, early stopping, and regularization.\n\n5. Regularization\nRegularization adds a penalty on large weights to the loss function so the model stays simple. L2 regularization (ridge) penalizes the sum of squared weights and shrinks them toward zero. L1 regularization (lasso) penalizes the sum of absolute weights and can set some weights exactly to zero, which performs feature selection. The strength of the penalty is controlled by a hyperparameter lambda; a larger lambda means stronger regularization.\n\n6. Evaluation metrics for classification\nAccuracy is the fraction of predictions that are correct, but it is misleading on imbalanced data. For example, if 99% of emails are not spam, a model that always predicts not spam has 99% accuracy but is useless. Precision is the fraction of predicted positives that are truly positive: TP / (TP + FP). Recall is the fraction of actual positives that the model finds: TP / (TP + FN). The F1 score is the harmonic mean of precision and recall. Use recall when missing a positive is costly, for example cancer screening, and precision when false alarms are costly, for example spam filters that block real email.\n"
SAMPLE_OUTPUT_JSON = "{\"overview\": \"Machine learning builds systems that learn patterns from data rather than following hand-written rules. The lecture covers supervised tasks, how to split data and avoid leakage, how overfitting and regularization control model complexity, and how to evaluate classifiers. Choosing the right data splits and metrics is what makes a model's reported performance trustworthy.\", \"big_picture\": \"Every topic serves one goal: building models that generalize to unseen data and measuring that honestly.\", \"topics\": [{\"title\": \"What is machine learning?\", \"bullets\": [\"Systems learn patterns from data instead of following hand-written rules.\", \"Three paradigms: supervised, unsupervised and reinforcement learning.\", \"Supervised uses labeled data; unsupervised finds structure without labels.\"], \"terms\": [{\"term\": \"Supervised learning\", \"definition\": \"Learning from labeled examples where every input comes with the correct answer.\", \"why_it_matters\": \"It is the main setting for the rest of the course.\", \"chunk_ids\": []}, {\"term\": \"Unsupervised learning\", \"definition\": \"Finding structure, such as clusters, in data that has no labels.\", \"why_it_matters\": \"Contrasts with supervised learning.\", \"chunk_ids\": []}]}, {\"title\": \"Supervised learning tasks\", \"bullets\": [\"Classification predicts a discrete category, e.g. spam or not spam.\", \"Regression predicts a continuous number, e.g. a house price.\"], \"terms\": [{\"term\": \"Classification\", \"definition\": \"Predicting a discrete category for an input.\", \"why_it_matters\": \"Determines which metrics apply.\", \"chunk_ids\": []}, {\"term\": \"Regression\", \"definition\": \"Predicting a continuous numeric value.\", \"why_it_matters\": \"Different loss and metrics from classification.\", \"chunk_ids\": []}]}, {\"title\": \"Train, validation and test sets\", \"bullets\": [\"Training set fits the model; validation set tunes it; test set is used once at the end.\", \"Reusing the test set causes data leakage and over-optimistic estimates.\", \"k-fold cross-validation helps when data is limited.\"], \"terms\": [{\"term\": \"Validation set\", \"definition\": \"Held-out data used to tune hyperparameters and choose between models.\", \"why_it_matters\": \"Keeps the test set untouched.\", \"chunk_ids\": []}, {\"term\": \"k-fold cross-validation\", \"definition\": \"Split data into k folds; train on k-1, validate on one, repeat k times and average.\", \"why_it_matters\": \"Gives reliable estimates with limited data.\", \"chunk_ids\": []}]}, {\"title\": \"Overfitting and underfitting\", \"bullets\": [\"Overfitting memorizes noise: low training error, high error on new data.\", \"Underfitting is too simple: both training and validation error are high.\", \"Fixes include more data, simpler models, early stopping, regularization.\"], \"terms\": [{\"term\": \"Overfitting\", \"definition\": \"When a model memorizes noise in the training data and generalizes poorly.\", \"why_it_matters\": \"The most common reason models fail in practice.\", \"chunk_ids\": []}, {\"term\": \"Bias-variance tradeoff\", \"definition\": \"High bias means too simple (underfit); high variance means too sensitive to training data (overfit).\", \"why_it_matters\": \"Frames how to choose model complexity.\", \"chunk_ids\": []}]}, {\"title\": \"Regularization\", \"bullets\": [\"A penalty on large weights keeps the model simple.\", \"L2 (ridge) shrinks weights; L1 (lasso) can zero them out for feature selection.\", \"Larger lambda means stronger regularization.\"], \"terms\": [{\"term\": \"L2 regularization (ridge)\", \"definition\": \"Penalizes the sum of squared weights, shrinking them toward zero.\", \"why_it_matters\": \"A standard way to reduce overfitting.\", \"chunk_ids\": []}, {\"term\": \"L1 regularization (lasso)\", \"definition\": \"Penalizes the sum of absolute weights and can set some exactly to zero.\", \"why_it_matters\": \"Performs feature selection.\", \"chunk_ids\": []}]}, {\"title\": \"Evaluation metrics for classification\", \"bullets\": [\"Accuracy misleads on imbalanced data.\", \"Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 is their harmonic mean.\", \"Prefer recall when misses are costly, precision when false alarms are costly.\"], \"terms\": [{\"term\": \"Precision\", \"definition\": \"Fraction of predicted positives that are truly positive.\", \"why_it_matters\": \"Matters when false alarms are costly.\", \"chunk_ids\": []}, {\"term\": \"Recall\", \"definition\": \"Fraction of actual positives the model finds.\", \"why_it_matters\": \"Matters when missing a positive is costly.\", \"chunk_ids\": []}, {\"term\": \"F1 score\", \"definition\": \"Harmonic mean of precision and recall.\", \"why_it_matters\": \"Single number balancing both.\", \"chunk_ids\": []}]}], \"questions\": [{\"topic\": \"What is machine learning?\", \"type\": \"mcq\", \"difficulty\": \"easy\", \"question\": \"Which learning paradigm uses labeled training data?\", \"options\": [\"Supervised learning\", \"Unsupervised learning\", \"Reinforcement learning\", \"Clustering\"], \"answer\": \"Supervised learning\", \"explanation\": \"Each supervised example comes with the correct answer.\", \"source_chunk_ids\": [], \"source_quote\": \"In supervised learning the training data is labeled\"}, {\"topic\": \"What is machine learning?\", \"type\": \"short\", \"difficulty\": \"medium\", \"question\": \"What is the goal of unsupervised learning?\", \"options\": [], \"answer\": \"To find structure in unlabeled data, for example by clustering customers into groups.\", \"explanation\": \"There are no labels, so the model looks for structure.\", \"source_chunk_ids\": [], \"source_quote\": \"there are no labels, and the goal is to find structure in the data\"}, {\"topic\": \"Supervised learning tasks\", \"type\": \"mcq\", \"difficulty\": \"easy\", \"question\": \"Predicting the price of a house is which kind of task?\", \"options\": [\"Classification\", \"Regression\", \"Clustering\", \"Reinforcement learning\"], \"answer\": \"Regression\", \"explanation\": \"Regression predicts a continuous number.\", \"source_chunk_ids\": [], \"source_quote\": \"Regression predicts a continuous number\"}, {\"topic\": \"Supervised learning tasks\", \"type\": \"application\", \"difficulty\": \"medium\", \"question\": \"A bank wants to label each transaction as fraudulent or legitimate. Is this classification or regression, and why?\", \"options\": [], \"answer\": \"Classification, because the output is a discrete category.\", \"explanation\": \"Fraud / not fraud is a category, like spam / not spam.\", \"source_chunk_ids\": [], \"source_quote\": \"Classification predicts a discrete category\"}, {\"topic\": \"Train, validation and test sets\", \"type\": \"mcq\", \"difficulty\": \"medium\", \"question\": \"What is the validation set used for?\", \"options\": [\"Fitting the model's weights\", \"Tuning hyperparameters and choosing between models\", \"Giving the final one-time performance estimate\", \"Increasing the amount of training data\"], \"answer\": \"Tuning hyperparameters and choosing between models\", \"explanation\": \"The test set, not the validation set, gives the final estimate.\", \"source_chunk_ids\": [], \"source_quote\": \"a validation set used to tune hyperparameters and choose between models\"}, {\"topic\": \"Train, validation and test sets\", \"type\": \"short\", \"difficulty\": \"medium\", \"question\": \"In 5-fold cross-validation, how many times is the model trained and what is used for validation each time?\", \"options\": [], \"answer\": \"Five times; each time one fold is held out for validation and the other four are used for training.\", \"explanation\": \"The process repeats k times and the scores are averaged.\", \"source_chunk_ids\": [], \"source_quote\": \"the model is trained on k-1 folds and validated on the remaining fold\"}, {\"topic\": \"Overfitting and underfitting\", \"type\": \"explain\", \"difficulty\": \"medium\", \"question\": \"Explain why a model with very low training error can still perform badly on new data.\", \"options\": [], \"answer\": \"It has overfit: it memorized noise in the training data instead of the real pattern.\", \"explanation\": \"Overfitting means low training error but high error on new data.\", \"source_chunk_ids\": [], \"source_quote\": \"so training error is low but error on new data is high\"}, {\"topic\": \"Overfitting and underfitting\", \"type\": \"mcq\", \"difficulty\": \"hard\", \"question\": \"Both training and validation error are high. What is the most likely diagnosis?\", \"options\": [\"Overfitting\", \"Underfitting\", \"Data leakage\", \"A good fit\"], \"answer\": \"Underfitting\", \"explanation\": \"The model is too simple to capture the pattern.\", \"source_chunk_ids\": [], \"source_quote\": \"so both training and validation error are high\"}, {\"topic\": \"Regularization\", \"type\": \"mcq\", \"difficulty\": \"medium\", \"question\": \"Which regularization method can set some weights exactly to zero?\", \"options\": [\"L2 (ridge)\", \"L1 (lasso)\", \"Early stopping\", \"Cross-validation\"], \"answer\": \"L1 (lasso)\", \"explanation\": \"This acts as feature selection.\", \"source_chunk_ids\": [], \"source_quote\": \"can set some weights exactly to zero, which performs feature selection\"}, {\"topic\": \"Regularization\", \"type\": \"application\", \"difficulty\": \"hard\", \"question\": \"A model is overfitting. You increase lambda. What effect do you expect and why?\", \"options\": [], \"answer\": \"Stronger regularization: weights are penalized more, the model becomes simpler, and overfitting should decrease.\", \"explanation\": \"Lambda controls the penalty strength.\", \"source_chunk_ids\": [], \"source_quote\": \"a larger lambda means stronger regularization\"}, {\"topic\": \"Evaluation metrics for classification\", \"type\": \"application\", \"difficulty\": \"hard\", \"question\": \"99% of emails are not spam and a model always predicts not spam. Why is its 99% accuracy misleading?\", \"options\": [], \"answer\": \"It never detects spam, so it is useless; accuracy is misleading on imbalanced data.\", \"explanation\": \"Accuracy hides the failure on the rare class.\", \"source_chunk_ids\": [], \"source_quote\": \"a model that always predicts not spam has 99% accuracy but is useless\"}, {\"topic\": \"Evaluation metrics for classification\", \"type\": \"mcq\", \"difficulty\": \"medium\", \"question\": \"Which metric should be prioritized for cancer screening?\", \"options\": [\"Precision\", \"Recall\", \"Accuracy\", \"Lambda\"], \"answer\": \"Recall\", \"explanation\": \"Missing a positive case is the costly error.\", \"source_chunk_ids\": [], \"source_quote\": \"Use recall when missing a positive is costly\"}], \"stats\": {\"note\": \"Hand-written example for demo mode; not LLM output.\"}}"

# ------------------------------------------------------------------ schemas ---


class TopicList(BaseModel):
    topics: list[str] = Field(description="3-8 short topic titles covering the lecture")


class Overview(BaseModel):
    overview: str = Field(description="Exactly 3 sentences summarising the lecture")
    big_picture: str = Field(description="One sentence on how the topics connect")


class TopicSummary(BaseModel):
    bullets: list[str] = Field(description="2-3 bullets, max 20 words each")


class KeyTerm(BaseModel):
    term: str
    definition: str = Field(description="Max 25 words, plain language")
    why_it_matters: str = Field(description="One short sentence")
    chunk_ids: list[int] = Field(description="Ids of the [chunk N] passages used")


class TermList(BaseModel):
    terms: list[KeyTerm] = Field(description="2-4 terms for this topic")


class Question(BaseModel):
    type: Literal["mcq", "short", "application", "explain"]
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    options: list[str] = Field(description="Exactly 4 options for mcq; empty list otherwise")
    answer: str
    explanation: str
    source_chunk_ids: list[int] = Field(description="Ids of the [chunk N] passages used")
    source_quote: str = Field(description="Short phrase copied VERBATIM from the notes")
    topic: str = Field(description="Title of the topic this question belongs to")


class QuestionList(BaseModel):
    questions: list[Question]


class Verdict(BaseModel):
    supported: bool = Field(description="Is the answer fully supported by the evidence?")
    single_correct_option: bool = Field(description="For MCQ: exactly one option is correct")
    issue: str = Field(description="Empty string if fine, else what is wrong")


class TopicBlock(BaseModel):
    title: str
    bullets: list[str]
    terms: list[KeyTerm]


class PackCore(BaseModel):
    overview: str
    big_picture: str
    topics: list[TopicBlock]
    questions: list[Question]


class StudyPack(PackCore):
    stats: dict = Field(default_factory=dict)


# ------------------------------------------------------------------ prompts ---

GROUNDING = ("Use ONLY the evidence passages provided. Each passage is tagged [chunk N]. "
             "Never add outside facts. If the evidence does not support something, leave it out.")



def _text_of(msg) -> str:
    c = getattr(msg, "content", msg)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (tolerates fences and extra prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in model reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


class StepChain:
    """One pipeline step. Tries native structured output (tool calling) first; if the model
    ignores the tool or returns nothing, falls back to asking for plain JSON and parsing it.
    This makes the app work across providers whose tool calling is flaky."""

    def __init__(self, llm, system: str, human: str, schema):
        self.schema = schema
        self.fallback_count = 0
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
        try:
            self.structured = prompt | llm.with_structured_output(schema)
        except Exception:
            self.structured = None
        json_prompt = ChatPromptTemplate.from_messages([
            ("system", system + " Respond with ONLY one JSON object, no prose and no markdown fences, "
                                "that matches this JSON schema: {schema_json}"),
            ("human", human)]).partial(schema_json=json.dumps(schema.model_json_schema()))
        self.json_chain = json_prompt | llm

    def invoke(self, variables: dict):
        if self.structured is not None:
            try:
                out = self.structured.invoke(variables)
                if out is not None:
                    return out
            except Exception as e:
                low = str(e).lower()
                if any(k in low for k in ("429", "quota", "rate limit", "resource_exhausted",
                                          "resource exhausted", "api key", "401", "403")):
                    raise                      # not a formatting problem: let the retry/backoff handle it
        self.fallback_count += 1
        reply = self.json_chain.invoke(variables)
        return self.schema.model_validate(extract_json(_text_of(reply)))


def build_chains(llm) -> dict:
    def make(system: str, human: str, schema):
        return StepChain(llm, system, human, schema)

    return {
        "topics": make("You are a careful note-processing assistant. " + GROUNDING,
                       "Identify the main topics in this lecture.\n\n<evidence>\n{context}\n</evidence>",
                       TopicList),
        "overview": make("You are an expert tutor writing for a student revising before an exam. " + GROUNDING,
                         "Topics: {topics}\n\n<evidence>\n{context}\n</evidence>", Overview),
        "summary": make("You are an expert tutor. " + GROUNDING,
                        "Summarise the topic '{topic}'.\n\n<evidence>\n{context}\n</evidence>", TopicSummary),
        "terms": make("You are a glossary writer. " + GROUNDING + " Prefer terms that are defined or emphasised.",
                      "Extract key terms for the topic '{topic}'. Cite chunk ids.\n\n"
                      "<evidence>\n{context}\n</evidence>", TermList),
        "questions": make(
            "You are an exam-writer. " + GROUNDING + " Every question must be answerable from the evidence "
            "alone. MCQs need 4 plausible options with exactly one correct. source_quote must be copied "
            "verbatim from the evidence. Cite source_chunk_ids.",
            "Write {n} practice questions on '{topic}', using these types in order: {type_hint}. "
            "Mix difficulty (roughly 30% easy, 50% medium, 20% hard). Set topic to '{topic}'.\n\n"
            "<evidence>\n{context}\n</evidence>", QuestionList),
        "verify": make("You are a strict fact-checker. " + GROUNDING,
                       "Question: {question}\nOptions: {options}\nProposed answer: {answer}\n"
                       "Explanation: {explanation}\n\n<evidence>\n{context}\n</evidence>", Verdict),
        "repair": make("You are an exam-writer fixing a flawed question. " + GROUNDING,
                       "This question failed review.\nQuestion: {question}\nProblem: {issue}\n\n"
                       "Write {n} corrected replacement question(s) on '{topic}'. Set topic to '{topic}'.\n\n"
                       "<evidence>\n{context}\n</evidence>", QuestionList),
        "baseline": make("You are an expert tutor and exam writer. Use only the lecture notes provided.",
                         "Create a complete study pack from these lecture notes: a 3-sentence overview, a "
                         "big-picture sentence, per-topic bullets and key terms, and {n} practice questions "
                         "(mix of mcq, short, application, explain; mixed difficulty). Each question needs a "
                         "source_quote copied verbatim from the notes.\n\n<notes>\n{notes}\n</notes>", PackCore),
    }


# ---------------------------------------------------------------- embeddings ---

_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were "
            "which with can not but if so we".split())


class HashEmbeddings(Embeddings):
    """Built-in embeddings: hashed bag-of-words. No download, no key, works offline.
    Good enough to retrieve the notes passages that mention a topic's words."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in _STOP:
                continue
            h = zlib.crc32(tok[:7].encode("utf-8"))      # crude stemming: first 7 characters
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embeddings(choice: str) -> Embeddings:
    if choice.startswith("Gemini"):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise RuntimeError("Missing package. Run: pip install langchain-google-genai") from e
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if choice.startswith("HuggingFace"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError("Missing package. Run: pip install langchain-huggingface "
                               "sentence-transformers") from e
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return HashEmbeddings()


def get_llm(provider: str, model: str):
    try:
        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, temperature=0)
        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, temperature=0)
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=0, num_ctx=8192)
    except ImportError as e:
        raise RuntimeError(f"Missing package for {provider}. Run: {PIP_HINT[provider]}") from e
    raise ValueError(f"Unknown provider: {provider}")


# ------------------------------------------------------------------ helpers ---


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"[chunk {d.metadata['chunk_id']}]\n{d.page_content}" for d in docs)


def quote_in_docs(quote: str, docs: list[Document]) -> bool:
    q = norm(quote)
    return bool(q) and any(q in norm(d.page_content) for d in docs)


TYPES = ["mcq", "short", "application", "explain"]
ProgressFn = Callable[[str, float], None]

# ---------------------------------------------------------------- generator ---


class StudyPackGenerator:
    def __init__(self, llm=None, embeddings=None, k: int = 5, retries: int = 4,
                 backoff: bool = True, chains: dict | None = None, min_interval: float = 0.0):
        self.chains = chains if chains is not None else build_chains(llm)
        self.embeddings = embeddings if embeddings is not None else HashEmbeddings()
        self.k, self.retries, self.backoff, self.min_interval = k, retries, backoff, min_interval
        self._last_call = 0.0
        self.chunks: list[Document] = []
        self.retriever = None
        self.reset_stats()

    def reset_stats(self):
        for c in getattr(self, "chains", {}).values():
            if hasattr(c, "fallback_count"):
                c.fallback_count = 0
        self.stats = {"llm_calls": 0, "attempts": 0, "failed_attempts": 0, "errors": [], "unverified": 0}

    def _call(self, name: str, variables: dict):
        """Invoke one chain with throttling and retries (rate limits, malformed output)."""
        self.stats["llm_calls"] += 1
        last = None
        for i in range(self.retries):
            self.stats["attempts"] += 1
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                out = self.chains[name].invoke(variables)
                if out is None:
                    raise ValueError("model returned empty structured output")
                return out
            except Exception as e:
                last = e
                self.stats["failed_attempts"] += 1
                msg = str(e).lower()
                limited = any(k in msg for k in ("429", "quota", "rate limit", "resource_exhausted",
                                                 "resource exhausted"))
                if self.backoff:
                    time.sleep(15 * (i + 1) if limited else min(2 ** i, 8))
        raise RuntimeError(f"step '{name}' failed after {self.retries} attempts: {last}")

    # 1. index
    def index(self, text: str, chunk_size: int = 900, overlap: int = 120):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        self.chunks = splitter.create_documents([text])
        for i, c in enumerate(self.chunks):
            c.metadata["chunk_id"] = i
        store = InMemoryVectorStore.from_documents(self.chunks, self.embeddings)
        self.retriever = store.as_retriever(search_type="mmr",
                                            search_kwargs={"k": self.k, "fetch_k": self.k * 3})
        return self.chunks

    def evidence(self, q: Question) -> list[Document]:
        ids = {i for i in q.source_chunk_ids if 0 <= i < len(self.chunks)}
        docs = [self.chunks[i] for i in sorted(ids)]
        for d in self.retriever.invoke(f"{q.question} {q.answer}"):
            if d.metadata["chunk_id"] not in ids:
                docs.append(d)
        return docs

    def _topic_sample(self, max_chunks: int = 12) -> list[Document]:
        if len(self.chunks) <= max_chunks:
            return self.chunks
        step = len(self.chunks) / max_chunks
        return [self.chunks[int(i * step)] for i in range(max_chunks)]

    # 2-3. generate
    def generate(self, text: str, questions_per_topic: int = 3,
                 progress: ProgressFn | None = None) -> StudyPack:
        report = progress or (lambda msg, frac: None)
        self.reset_stats()
        report("Indexing notes", 0.05)
        self.index(text)
        sample_ctx = fmt(self._topic_sample())
        report("Discovering topics", 0.12)
        topics = self._call("topics", {"context": sample_ctx}).topics[:8]
        overview = self._call("overview", {"topics": ", ".join(topics), "context": sample_ctx})

        blocks: list[TopicBlock] = []
        raw: list[Question] = []
        for i, topic in enumerate(topics):
            report(f"Topic {i + 1}/{len(topics)}: {topic}", 0.2 + 0.55 * i / max(1, len(topics)))
            docs = self.retriever.invoke(topic)
            ctx = fmt(docs)
            hint = ", ".join(TYPES[(i + j) % 4] for j in range(questions_per_topic))
            try:
                summary = self._call("summary", {"topic": topic, "context": ctx})
                terms = self._call("terms", {"topic": topic, "context": ctx})
                qs = self._call("questions", {"topic": topic, "context": ctx, "n": questions_per_topic,
                                              "type_hint": hint}).questions
            except RuntimeError as e:
                self.stats["errors"].append(str(e))
                continue
            blocks.append(TopicBlock(title=topic, bullets=summary.bullets, terms=terms.terms))
            for q in qs:
                q.topic = topic
                raw.append(q)

        if not blocks:
            raise RuntimeError("No topic could be generated. " + (self.stats["errors"][0]
                                                                  if self.stats["errors"] else ""))
        report("Verifying questions against the notes", 0.8)
        final = self._verify_and_repair(raw, report)
        report("Done", 1.0)
        self.stats["json_fallbacks"] = sum(getattr(c, "fallback_count", 0) for c in self.chains.values())
        return StudyPack(overview=overview.overview, big_picture=overview.big_picture,
                         topics=blocks, questions=final, stats=dict(self.stats))

    # 4. verify
    def _check(self, q: Question) -> str:
        docs = self.evidence(q)
        if not quote_in_docs(q.source_quote, docs):
            return "source_quote not found verbatim in the notes"
        try:
            v = self._call("verify", {"question": q.question, "options": q.options or "n/a",
                                      "answer": q.answer, "explanation": q.explanation,
                                      "context": fmt(docs)})
        except RuntimeError:
            self.stats["unverified"] += 1       # verifier unavailable: keep, flag in stats
            return ""
        if not v.supported:
            return v.issue or "answer not supported by the notes"
        if q.type == "mcq" and not v.single_correct_option:
            return v.issue or "MCQ does not have exactly one correct option"
        return ""

    def _verify_and_repair(self, questions: list[Question], report: ProgressFn) -> list[Question]:
        kept: list[Question] = []
        failed: list[tuple[Question, str]] = []
        for n, q in enumerate(questions):
            report(f"Verifying question {n + 1}/{len(questions)}", 0.8 + 0.12 * n / max(1, len(questions)))
            issue = self._check(q)
            if issue:
                failed.append((q, issue))
            else:
                kept.append(q)
        repaired = dropped = 0
        for q, issue in failed:
            try:
                new = self._call("repair", {"question": q.question, "issue": issue, "topic": q.topic,
                                            "n": 1, "context": fmt(self.evidence(q))}).questions
            except RuntimeError:
                new = []
            ok = False
            for nq in new:
                nq.topic = q.topic
                if not self._check(nq):
                    kept.append(nq)
                    ok = True
            repaired += ok
            dropped += not ok
        self.stats.update(questions_generated=len(questions), questions_failed_first_check=len(failed),
                          questions_repaired=repaired, questions_dropped=dropped)
        return kept

    def baseline(self, text: str, n_questions: int = 12) -> StudyPack:
        self.reset_stats()
        core = self._call("baseline", {"notes": text, "n": n_questions})
        return StudyPack(**core.model_dump(), stats=dict(self.stats))


# ------------------------------------------------------------------ exports ---


def to_markdown(pack: StudyPack) -> str:
    out = ["# Study Pack\n", "## Overview", pack.overview, f"\n**Big picture:** {pack.big_picture}\n",
           "## Topic Summaries"]
    for t in pack.topics:
        out.append(f"\n### {t.title}")
        out += [f"- {b}" for b in t.bullets]
    out += ["\n## Key Terms", "| Term | Definition | Why it matters |", "|---|---|---|"]
    for t in pack.topics:
        for k in t.terms:
            out.append(f"| **{k.term}** | {k.definition} | {k.why_it_matters} |")
    out.append("\n## Practice Questions")
    for n, q in enumerate(pack.questions, 1):
        out.append(f"\n**Q{n}. [{q.difficulty} / {q.type}] {q.question}**")
        out += [f"   - {l}. {o}" for l, o in zip("ABCD", q.options)]
    out.append("\n## Answer Key")
    for n, q in enumerate(pack.questions, 1):
        out.append(f"\n**Q{n}.** {q.answer}\n> {q.explanation}\n> _Source: \"{q.source_quote}\"_")
    return "\n".join(out)


def to_anki_csv(pack: StudyPack) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["front", "back"])
    for t in pack.topics:
        for k in t.terms:
            w.writerow([k.term, k.definition])
    for q in pack.questions:
        front = q.question
        if q.options:
            front += " | " + " / ".join(f"{l}. {o}" for l, o in zip("ABCD", q.options))
        w.writerow([front, f"{q.answer} - {q.explanation}"])
    return buf.getvalue()


# ----------------------------------------------------------------------- UI ---

EMBED_CHOICES = ["Built-in (no download, recommended)", "Gemini embeddings (uses your Gemini key)",
                 "HuggingFace MiniLM (needs torch, large download)"]


def preflight(provider: str) -> str:
    if provider in KEY_ENV and not os.environ.get(KEY_ENV[provider], "").strip():
        return f"No API key for {provider}. Paste it in the sidebar or at the top of the file."
    if provider == "ollama":
        try:
            socket.create_connection(("localhost", 11434), timeout=2).close()
        except OSError:
            return ("Ollama is not running on this computer. Start it, or choose gemini or groq "
                    "in the sidebar and enter a free API key.")
    return ""


def render_pack(pack: StudyPack):
    s = pack.stats or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics", len(pack.topics))
    c2.metric("Key terms", sum(len(t.terms) for t in pack.topics))
    c3.metric("Questions", len(pack.questions))
    c4.metric("Dropped by verifier", s.get("questions_dropped", "-"))

    tab_sum, tab_terms, tab_quiz, tab_verify, tab_dl = st.tabs(
        ["Summary", "Key terms", "Practice quiz", "Grounding & verification", "Download"])
    with tab_sum:
        st.subheader("Overview")
        st.write(pack.overview)
        st.info(f"**Big picture:** {pack.big_picture}")
        for t in pack.topics:
            with st.expander(t.title, expanded=True):
                for b in t.bullets:
                    st.markdown(f"- {b}")
    with tab_terms:
        rows = [{"Term": k.term, "Definition": k.definition, "Why it matters": k.why_it_matters,
                 "Topic": t.title} for t in pack.topics for k in t.terms]
        st.dataframe(rows, hide_index=True)
    with tab_quiz:
        for n, q in enumerate(pack.questions, 1):
            st.markdown(f"**Q{n}. {q.question}**  \n`{q.type}` | `{q.difficulty}` | _{q.topic}_")
            for letter, opt in zip("ABCD", q.options):
                st.markdown(f"&nbsp;&nbsp;&nbsp;**{letter}.** {opt}")
            with st.expander("Show answer"):
                st.success(q.answer)
                st.write(q.explanation)
                st.caption(f'Source quote from your notes: "{q.source_quote}"')
    with tab_verify:
        st.write("Every question carries a quote copied from the notes. The pipeline checks the quote "
                 "exists verbatim, then re-retrieves evidence and asks a verifier LLM whether the answer "
                 "is supported. Failures are repaired once, then dropped.")
        st.json(s)
    with tab_dl:
        st.download_button("Study pack (.md)", to_markdown(pack), "study_pack.md", "text/markdown")
        st.download_button("Flashcards (.csv)", to_anki_csv(pack), "flashcards.csv", "text/csv")
        st.download_button("Raw data (.json)", pack.model_dump_json(indent=2), "study_pack.json",
                           "application/json")


@st.cache_resource(show_spinner="Preparing embeddings...")
def cached_embeddings(choice: str):
    return get_embeddings(choice)


def main():
    st.set_page_config(page_title="Lecture Notes -> Study Pack", layout="wide")
    key_code = {"gemini": GEMINI_API_KEY, "groq": GROQ_API_KEY}

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("Free LLM provider", list(DEFAULT_MODELS),
                                help="gemini / groq need a free API key. ollama = local model.")
        model = st.text_input("Model", DEFAULT_MODELS[provider], key=f"model_{provider}")
        if provider in KEY_ENV:
            env = KEY_ENV[provider]
            key = st.text_input(env, type="password", value=key_code[provider] or os.getenv(env, ""))
            if key.strip():
                os.environ[env] = key.strip()
        else:
            st.caption("Ollama must be installed and running: `ollama pull llama3.1:8b`")
        emb_choice = st.selectbox("Embeddings for retrieval", EMBED_CHOICES)
        qpt = st.slider("Questions per topic", 1, 5, 2)

        if st.button("Test LLM connection"):
            problem = preflight(provider)
            if problem:
                st.error(problem)
            else:
                try:
                    reply = get_llm(provider, model).invoke("Reply with the single word OK.")
                    st.success(f"Connected. Model replied: {str(reply.content)[:60]}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
                    low = str(e).lower()
                    if "model" in low and any(k in low for k in ("not exist", "not found", "404", "no access")):
                        st.info("The model name is retired or unavailable. Change the Model box to a current "
                                "free model (see console.groq.com/docs/models or aistudio.google.com).")
        st.divider()
        if st.button("Show example output (no LLM needed)"):
            st.session_state["pack"] = StudyPack(**json.loads(SAMPLE_OUTPUT_JSON))
            st.session_state["example"] = True

    st.title("Lecture Notes -> Study Pack")
    st.caption("Summaries, key terms and practice questions grounded in your notes with RAG, "
               "using a free LLM and a verification step.")

    source = st.radio("Notes source", ["Sample notes", "Paste text", "Upload file (.txt / .md / .pdf)"],
                      horizontal=True)
    text = ""
    if source == "Sample notes":
        text = SAMPLE_NOTES
        with st.expander("Preview sample notes"):
            st.text(text)
    elif source == "Paste text":
        text = st.text_area("Paste lecture notes", height=250)
    else:
        up = st.file_uploader("Upload notes", type=["txt", "md", "pdf"])
        if up is not None:
            if up.name.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    text = "\n".join(p.extract_text() or "" for p in PdfReader(up).pages)
                except ImportError:
                    st.error("PDF support needs: pip install pypdf")
            else:
                text = up.read().decode("utf-8", errors="ignore")

    if st.button("Generate study pack", type="primary", disabled=not text.strip()):
        problem = preflight(provider)
        if problem:
            st.error(problem)
            st.stop()
        bar, status = st.progress(0.0), st.empty()
        try:
            gen = StudyPackGenerator(get_llm(provider, model), cached_embeddings(emb_choice),
                                     min_interval=float(os.getenv("STUDY_MIN_INTERVAL",
                                                                  MIN_INTERVAL[provider])))
            pack = gen.generate(text, qpt, progress=lambda m, f: (bar.progress(min(f, 1.0)),
                                                                  status.write(m)))
            st.session_state["pack"], st.session_state["example"] = pack, False
            status.empty()
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.caption("Use 'Test LLM connection' in the sidebar to check the key and model name. "
                       "Free tiers allow few requests per minute: a full run takes a few minutes, and "
                       "'quota' or '429' errors mean wait a minute and retry.")

    if "pack" in st.session_state:
        if st.session_state.get("example"):
            st.warning("Showing a pre-written example (no LLM was used).")
        render_pack(st.session_state["pack"])


if __name__ == "__main__":
    main()
