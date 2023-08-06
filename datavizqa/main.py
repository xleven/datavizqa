import os
import re
from operator import itemgetter, attrgetter

import streamlit as st
import pandas as pd

import langchain
from langchain.agents import AgentType, AgentExecutor, OpenAIFunctionsAgent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import OutputFunctionsParser, JsonKeyOutputFunctionsParser
from langchain.tools import format_tool_to_openai_function, PythonAstREPLTool
from langchain.schema import FunctionMessage, HumanMessage, SystemMessage, AgentFinish, AgentAction
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableMap, RunnablePassthrough, RunnableLambda, RouterRunnable

from tools import PythonPlotTool


langchain.debug = True
RE_MARKDOWN_IMAGE = r"!\[(.*?)\]\((.*?)\)"


st.set_page_config(page_title="DataVizQA", page_icon="🤖")
st.title("QA on your data with visualizations")

ss = st.session_state

with st.sidebar:
    ss.openai_api_key = st.text_input("Your OpenAI API key", placeholder="sk-xxxx")
    ss.cot = st.radio(
        "Expand new thoughts", [False, True], format_func=lambda x: "Yes" if x else "No")


@st.cache_data
def load_csv(csv) -> pd.DataFrame:
    return pd.read_csv(csv)


def is_function_call(fn_llm):
    if fn_llm.additional_kwargs.get("function_call"):
        return "is_function_call"
    else:
        return "no_function_call"

def get_args_format(fn_llm):
    if fn_llm.additional_kwargs["function_call"]["arguments"].startswith("{"):
        return "json"
    else:
        return "text"

def get_chain(df, number_of_head_rows=5, openai_api_key=""):
    SYSTEM_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_head}""".format(df_head=str(df.head(number_of_head_rows).to_markdown()))

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        openai_api_key=openai_api_key,
        temperature=0,
        streaming=True,
    )
    tools = [
        PythonPlotTool(name="python_plot", locals={"df": df}, outdir="./datavizqa/static/"),
        PythonAstREPLTool(name="python", locals={"df": df}),
    ]
    fn_list = [format_tool_to_openai_function(tool) for tool in tools]
    fn_dict = {tool.name: tool for tool in tools}

    parse_chain = (
        {
            "key": itemgetter("fn_llm") | RunnableLambda(get_args_format),
            "input": itemgetter("fn_llm") | RunnablePassthrough(),
        }
        | RouterRunnable({
            "json": JsonKeyOutputFunctionsParser(key_name="__arg1"),
            "text": OutputFunctionsParser(),
        })
    )

    fn_chain = (
        RunnableMap({
            "history": lambda x: prompt.format_messages(question=x["question"]),
            "ai_message": itemgetter("fn_llm"),
            "fn_message": (
                RunnableMap({
                    "fn_llm": itemgetter("fn_llm"),
                    "fn_name": lambda x: x["fn_llm"].additional_kwargs["function_call"]["name"],
                    "fn_in": parse_chain,
                })
                | {
                    "fn_name": itemgetter("fn_name"),
                    "fn_out":  lambda x: fn_dict[x["fn_name"]](x["fn_in"])
                }
                | (lambda x: FunctionMessage(name=x["fn_name"], content=x["fn_out"]))
            )
        })
        | (lambda x: x["history"] + [x["ai_message"], x["fn_message"]])
        | llm
    )

    main_chain = (
        RunnableMap({
            "question": itemgetter("question"),
            "fn_llm": prompt | llm.bind(functions=fn_list)
        })
        | {
            "key": itemgetter("fn_llm") | RunnableLambda(is_function_call),
            "input": RunnablePassthrough(),
        }
        | RouterRunnable({
            "no_function_call": itemgetter("fn_llm") | StrOutputParser(),
            "is_function_call": fn_chain,
        })
    )
    return main_chain


csv = st.file_uploader("Upload your CSV file", type=["csv"])
if csv is not None:
    df = load_csv(csv)
    st.dataframe(df.head())
    if key := ss.openai_api_key or os.getenv("OPENAI_API_KEY"):
        ss.chain = get_chain(df, openai_api_key=key)

if "chain" in ss:
    if "messages" not in ss:
        ss.messages = [{"role": "assistant", "content": "Data loaded! Ask me anything! I can also plot charts!"}]
    for message in ss.messages:
        st.chat_message(message["role"]).write(message["content"])

    if question := st.chat_input(placeholder="Your question"):
        ss.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            answer, container = "", st.empty()
            for token in ss.chain.stream({"question": question}):
                answer += token.content
                container.write(answer)
        ss.messages.append({"role": "assistant", "content": answer})