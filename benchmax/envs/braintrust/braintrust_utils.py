import requests
from typing import Any, Dict, List
from dataclasses import asdict
from benchmax.envs.types import ToolDefinition

def get_functions(braintrust_api_key: str, braintrust_project_id: str) -> Dict:
    ''' Returns a function dictionary that contains tools, prompts, and scorers from function API call
            self.functions["tools"]: Dict[str, Tuple[ToolDefinition(now a dict), Callable]]
            self.functions["prompts"]: Dict[str, Dict]
            self.functions["scorers"]: Dict[str, Dict]
    '''
    url = f"https://api.braintrust.dev/v1/function?project_id={braintrust_project_id}"
    headers = {
    "Authorization": f"Bearer {braintrust_api_key}"
    }
    response = requests.request("GET", url, headers=headers)
    
    try:
        data = response.json()
        tools, prompts, scorers = {}, {}, {}
        for object in data["objects"]:
            if object["function_type"] == "tool": 
                tool_definition = asdict(ToolDefinition(
                    name=object["name"],
                    description=object["description"],
                    input_schema={
                        "type":"object",
                        "properties": object["function_schema"]["parameters"]["properties"],
                        "required": object["function_schema"]["parameters"]["required"]
                    }
                )) # Converted to dict to allow JSON serializing. 
                tool_callable = object["function_data"]["data"]["preview"]
                tools[object["id"]] = (tool_definition, tool_callable)
            elif object["function_type"] == "scorer":
                scorers[object["id"]] = object["function_data"]
            elif not object["function_type"] and object["prompt_data"]:
                prompts[object["id"]] = object["prompt_data"]["prompt"]["messages"]

        return {"tools": tools, "prompts": prompts, "scorers": scorers}
    except ValueError:
        print("Response is not valid JSON:")
        print(response.text)
        raise Exception("Failed to get functions(tools, scorers, and prompts)")

def get_dataset_with_id(braintrust_api_key: str, dataset_id: str) -> List[Dict[str, str]]:
    # returns dataset given a dataset_id
    url = f"https://api.braintrust.dev/v1/dataset/{dataset_id}/fetch"
    json = {
    }
    headers = {
    "Authorization": f"Bearer {braintrust_api_key}"
    }
    response = requests.request("POST", url, json=json, headers=headers)
    
    try:
        data = response.json()
        processed_data = []
        print("DEBUG: data =", data)
        for event in data["events"]:
            processed_data.append({"prompt": event["input"], "ground_truth": event["expected"]})
        return processed_data
    except ValueError:
        print("Response is not valid JSON:")
        print(response.text)
        raise Exception(f"Failed to get dataset with id:{dataset_id}")

def get_dataset_ids(braintrust_api_key: str, braintrust_project_id: str) -> Dict[str, str]:
    # returns dataset ids of project
    url = f"https://api.braintrust.dev/v1/dataset?project_id={braintrust_project_id}"
    headers = {
    "Authorization": f"Bearer {braintrust_api_key}"
    }
    response = requests.request("GET", url, headers=headers)
    
    try:
        dataset_objects = response.json()
        dataset_ids = dict([(dataset_objects["objects"][i]["id"], dataset_objects["objects"][i]["name"]) for i in range(len(dataset_objects["objects"]))])
        return dataset_ids
    except ValueError:
        print("Response is not valid JSON:")
        print(response.text)
        raise Exception("Failed to get datasets")

def get_project_data(braintrust_api_key: str, braintrust_project_id: str) -> Dict:
    # returns project data
    url = f"https://api.braintrust.dev/v1/project?limit=1&ids={braintrust_project_id}"
    headers = {
    "Authorization": f"Bearer {braintrust_api_key}"
    }
    response = requests.request("GET", url, headers=headers)
    
    try:
        return response.json()
    except ValueError:
        print("Response is not valid JSON:")
        print(response.text)
        raise Exception("Failed to get project data")