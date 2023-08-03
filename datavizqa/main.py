import os
import re

import streamlit as st
import pandas as pd

import langchain
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonAstREPLTool
from langchain.schema import SystemMessage

from tools import PythonPlotTool
from agents import OpenAIFunctionsAgentFix  # https://github.com/langchain-ai/langchain/issues/6364


@st.cache_data
def load_csv(csv) -> pd.DataFrame:
    return pd.read_csv(csv)


def get_agent(df, openai_api_key, number_of_head_rows=5, outdir="./datavizqa/static"):
    SYSTEM_PROMPT = """You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_head}""".format(df_head=str(df.head(number_of_head_rows).to_markdown()))

    tools = [
        PythonPlotTool(locals={"df": df}, outdir=outdir),
        PythonAstREPLTool(name="python", locals={"df": df}),
    ]
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        openai_api_key=openai_api_key,
        temperature=0,
        streaming=True,
    )
    agent = OpenAIFunctionsAgentFix.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=SystemMessage(content=SYSTEM_PROMPT),
    )
    agent_exe = AgentExecutor.from_agent_and_tools(agent, tools)
    return agent_exe

langchain.debug = True

RE_MARKDOWN_IMAGE = r"!\[(.*?)\]\((.*?)\)"


st.set_page_config(page_title="DataVizQA", page_icon="🤖")
st.title("QA on your data with visualizations")

ss = st.session_state

with st.sidebar:
    ss.openai_api_key = st.text_input("Your OpenAI API key", placeholder="sk-xxxx")
    ss.cot = st.radio(
        "Expand new thoughts", [False, True], format_func=lambda x: "Yes" if x else "No")


csv = st.file_uploader("Upload your CSV file", type=["csv"])
if csv is not None:
    df = load_csv(csv)
    st.dataframe(df.head())
    if key := ss.openai_api_key or os.getenv("OPENAI_API_KEY"):
        ss.agent = get_agent(df, openai_api_key=key)

if "agent" in ss:
    if "messages" not in ss:
        ss.messages = [{"role": "assistant", "content": "Data loaded! Ask me anything! I can also plot charts!"}]
    for message in ss.messages:
        st.chat_message(message["role"]).write(message["content"])

    if question := st.chat_input(placeholder="Your question"):
        ss.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)
        with st.chat_message("assistant"):
            handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=ss.cot)
            output_image = ""
            for step in ss.agent.iter(question, callbacks=[handler]):
                if output := step.get("intermediate_step"):
                    action, value = output[0]
                    if action.tool == "python_plot":
                        output_image = value

            answer = step.get("output")
            if output_image:
                if re.search(RE_MARKDOWN_IMAGE, answer):
                    answer = re.sub(RE_MARKDOWN_IMAGE, f"![\g<1>]({output_image})", answer)
                else:
                    answer = answer + "\n" + f"![{output_image.split('/')[0]}]({output_image})"

            ss.messages.append({"role": "assistant", "content": answer})
            st.write(answer)