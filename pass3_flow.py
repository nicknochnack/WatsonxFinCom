import os
from dotenv import load_dotenv
from colorama import Fore, init
from langchain_ibm import WatsonxLLM
from TM1py.Services import TM1Service
from TM1py.Utils.Utils import build_pandas_dataframe_from_cellset

load_dotenv()
init()

parameters = {"decoding_method": "greedy", "max_new_tokens": 400}
llm = WatsonxLLM(
    model_id="meta-llama/llama-3-405b-instruct",
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
        cube_name="General Ledger", view_name="Pass 3_Input", private=False
    )
    df = build_pandas_dataframe_from_cellset(data, multiindex=True)


def template(row, total_variance):
    rows = ""
    for key, value in row.items():
        rows += f"{key}:{value} \n"

    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful management accounting analyst. Focused on providing  example concise explanations for the variance between forecast and actual results based on provided data.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    
    {rows}

    The total variance to explain is {total_variance}
    
    Consider these points:
    - The row which contains the Amount measure shows the difference between forecast and actual
    - The row which contains the Watsonx Commentary measure shows the explanation for the variance
    - A positive value indicates performance above forecast
    - A negative value indicates performance below forecast
    - 'Values' refers to the metric being measured
    - 'Name' refers to the specific accounting line item (e.g., department, division, account group/number)
    - Only provide the commentary, do not prefix it with 'The following is an example of a synthetic statement:', 'Answer:', or 'Here is the code I have so:'

    Provide a brief statement explaining the likely reasons for this variance, the explained variance should be the sum of the values.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>"""


accounts = list(set(df.index.get_level_values("Account")))

cellset = {}
total_variance = (
    df.xs("Amount", level="General Ledger Measure")
    .groupby("Sandboxes")
    .sum()["Values"]["Base"]
)
prompt = template(df.to_dict()["Values"], total_variance)
print(Fore.GREEN + prompt)
response = llm.invoke(prompt)
print(Fore.MAGENTA + response)
print("\n\n")
write_target = list(df.iloc[0].name)[:-1]
write_target[4] = "All Cost Centres"
write_target[5] = "All Accounts"
write_target.append("Watsonx Commentary")
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
