import os
from dotenv import load_dotenv
from colorama import Fore, init
from langchain_ibm import WatsonxLLM
from TM1py.Services import TM1Service
from TM1py.Utils.Utils import build_pandas_dataframe_from_cellset

load_dotenv()
init()

parameters = {"decoding_method": "greedy", "max_new_tokens": 100}
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-70b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id=os.environ["WATSONX_PROJECT_ID"],
)

# Import data from PA
with TM1Service(
    address=os.environ["TM1_ADDRESS"],
    port=os.environ["TM1_PORT"],
    user=os.environ["TM1_USER"],
    password=os.environ["TM1_PASSWORD"],
    ssl=os.environ["TM1_SSL"],
) as tm1:
    data = tm1.cubes.cells.execute_view(
        cube_name="General Ledger", view_name="Pass 1", private=False
    )
    df = build_pandas_dataframe_from_cellset(data, multiindex=True)
    df.to_csv("pass1.csv")


def template(row):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful management accounting analyst. Focused on providing  example concise explanations for the variance between forecast and actual results based on provided data. <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {row}

    Consider these points:
    - A positive value indicates performance above forecast
    - A negative value indicates performance below forecast
    - 'Values' refers to the metric being measured
    - 'Name' refers to the specific accounting line item (e.g., department, division, account group/number)
    - Only provide the commentary, do not prefix it with 'The following is an example of a synthetic statement:', 'Answer:', or 'Here is the code I have so:'

    Provide a brief synthetic statement explaining the likely reasons for this variance by speculating, this is just for demonstration so feel free to make it up.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""


cellset = {}
for index, row in df.iterrows():
    prompt = template(row)
    print(Fore.GREEN + prompt)
    response = llm.invoke(prompt)
    print(Fore.MAGENTA + response)
    print("\n\n")
    write_target = list(index)
    write_target[-1] = "Watsonx Commentary"
    del write_target[0]
    write_target = tuple(write_target)
    cellset[write_target] = response

with TM1Service(
    address=os.environ["TM1_ADDRESS"],
    port=os.environ["TM1_PORT"],
    user=os.environ["TM1_USER"],
    password=os.environ["TM1_PASSWORD"],
    ssl=os.environ["TM1_SSL"],
) as tm1:
    tm1.cubes.cells.write_values("General Ledger", cellset)
