from fastmcp import FastMCP
from simple_salesforce.api import Salesforce

from typing import Dict, List, Union, Optional, Literal, Annotated
from pydantic import Field
import ast
from datetime import datetime
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesforceConnector:
    def __init__(self, config: Dict):
        if "username" in config:
            self.sf = Salesforce(
                username=config["username"],
                password=config["password"],
                security_token=config["security_token"]
            )
        elif "instance_url" in config:
            self.sf = Salesforce(
                instance_url=config["instance_url"],
                session_id=config["session_id"]
            )
        else:
            raise ValueError("Invalid Salesforce configuration")

    def run_query(self, query: str) -> tuple:
        is_sosl = query.startswith("FIND")
        try:
            if not is_sosl:
                result = self.sf.query_all(query)
            else:
                result = self.sf.search(query)
        except Exception as e:
            e = str(e)
            e = ast.literal_eval(e.split("Response content:")[1].strip())[0]
            err = f"{e['errorCode']}: {e['message']}"
            return err, 0

        if not is_sosl:
            result_data = result["records"]
        else:
            result_data = result["searchRecords"]

        if len(result_data) == 0:
            return [], 1

        # Clean up results by removing attributes and null fields
        for row in result_data:
            if "attributes" in row:
                del row["attributes"]

        keys = result_data[0].keys()
        all_none_keys = [key for key in keys if all([record[key] is None for record in result_data])]
        new_data = [{k: v for k, v in record.items() if k not in all_none_keys} for record in result_data]
        
        return new_data, 1

# Initialize FastMCP server
mcp = FastMCP(
    name="SalesforceAPI",
    instructions="This server provides tools for interacting with Salesforce data through a REST API."
)

# Global connector instance
sf_connector = None

def set_salesforce_connector(config: Optional[Dict] = None, connector: Optional[SalesforceConnector] = None) -> SalesforceConnector:
    """Set the global Salesforce connector instance.
    
    Args:
        config (Dict, optional): Configuration dict with username, password, and security_token
        connector (SalesforceConnector, optional): Existing connector instance
        
    Returns:
        SalesforceConnector: The configured connector instance
        
    Raises:
        ValueError: If neither config nor connector is provided
    """
    global sf_connector
    if connector:
        sf_connector = connector
    elif config:
        sf_connector = SalesforceConnector(config)
    else:
        raise ValueError("Either config or connector must be provided")
    return sf_connector

@mcp.tool
def get_cases(
    start_date: Annotated[Optional[str], Field(description="Start date in format 'YYYY-MM-DDTHH:MM:SSZ'")] = None,
    end_date: Annotated[Optional[str], Field(description="End date in format 'YYYY-MM-DDTHH:MM:SSZ'")] = None,
    agent_ids: Annotated[Optional[List[str]], Field(description="List of agent IDs to filter by")] = None,
    case_ids: Annotated[Optional[List[str]], Field(description="List of case IDs to filter by")] = None,
    order_item_ids: Annotated[Optional[List[str]], Field(description="List of order item IDs to filter by")] = None,
    issue_ids: Annotated[Optional[List[str]], Field(description="List of issue IDs to filter by")] = None,
    statuses: Annotated[Optional[List[str]], Field(description="List of case statuses to filter by")] = None
) -> Union[List[Dict], str]:
    """Retrieve cases based on various filtering criteria."""
    query = """
        SELECT OwnerId, CreatedDate, ClosedDate, AccountId
        FROM Case
    """
    
    conditions = []
    if start_date:
        try:
            datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
            conditions.append(f"CreatedDate >= {start_date}")
        except ValueError:
            return "Error: start_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"
    
    if end_date:
        try:
            datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
            conditions.append(f"CreatedDate < {end_date}")
        except ValueError:
            return "Error: end_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"
    
    if agent_ids:
        if not isinstance(agent_ids, list):
            return "Error: agent_ids must be a list"
        conditions.append(f"OwnerId IN {tuple(agent_ids)}" if len(agent_ids) > 1 else f"OwnerId = '{agent_ids[0]}'")

    if case_ids:
        if not isinstance(case_ids, list):
            return "Error: case_ids must be a list"
        conditions.append(f"Id IN {tuple(case_ids)}" if len(case_ids) > 1 else f"Id = '{case_ids[0]}'")

    if order_item_ids:
        if not isinstance(order_item_ids, list):
            return "Error: order_item_ids must be a list"
        conditions.append(f"OrderItemId__c IN {tuple(order_item_ids)}" if len(order_item_ids) > 1 else f"OrderItemId__c = '{order_item_ids[0]}'")

    if issue_ids:
        if not isinstance(issue_ids, list):
            return "Error: issue_ids must be a list"
        conditions.append(f"IssueId__c IN {tuple(issue_ids)}" if len(issue_ids) > 1 else f"IssueId__c = '{issue_ids[0]}'")

    if statuses:
        if not isinstance(statuses, list):
            return "Error: statuses must be a list"
        conditions.append(f"Status IN {tuple(statuses)}" if len(statuses) > 1 else f"Status = '{statuses[0]}'")

    if conditions:
        query = f"{query} WHERE {' AND '.join(conditions)}"

    if not sf_connector:
        return "Error: Salesforce connection not configured"

    result, status = sf_connector.run_query(query)
    return result if status == 1 else result

@mcp.tool
def search_knowledge_articles(
    search_term: Annotated[str, Field(description="The term to search for in knowledge articles")]
) -> Union[List[Dict], str]:
    """Search for knowledge articles based on a given search term."""
    if not isinstance(search_term, str):
        return "Error: search_term must be a string"

    if not search_term.strip():
        return "Error: search_term cannot be empty"

    sosl_query = f"""
        FIND {{{search_term}}} IN ALL FIELDS 
        RETURNING Knowledge__kav(Id, Title, FAQ_Answer__c 
        WHERE PublishStatus='Online' AND Language='en_US')
    """.strip()

    if not sf_connector:
        return "Error: Salesforce connection not configured"

    result, status = sf_connector.run_query(sosl_query)
    if status == 0:
        return result

    articles = []
    for record in result:
        if isinstance(record, dict):
            articles.append({
                'Id': record.get('Id', ''),
                'Title': record.get('Title', ''),
                'Content': record.get('FAQ_Answer__c', '')
            })

    return articles

@mcp.tool
def get_start_date(
    end_date: Annotated[str, Field(description="The end date in ISO format 'YYYY-MM-DDTHH:MM:SSZ'")],
    period: Annotated[Literal["day", "week", "month", "quarter"], Field(description="The time period unit")],
    interval_count: Annotated[int, Field(description="The number of periods to subtract from the end date")]
) -> Union[str, str]:
    """Calculate the start date based on the end date, period, and interval count."""
    try:
        if not isinstance(end_date, str):
            return "Error: end_date must be a string"
        if not isinstance(period, str):
            return "Error: period must be a string"
        if not isinstance(interval_count, int):
            return "Error: interval_count must be an integer"
        
        if period not in ['day', 'week', 'month', 'quarter']:
            return "Error: Invalid period. Must be 'day', 'week', 'month', or 'quarter'"
        
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return "Error: Invalid end_date format. Expected format: 'YYYY-MM-DDTHH:MM:SSZ'"
        
        if interval_count < 0:
            return "Error: interval_count must be a non-negative integer"
        
        if period == 'week':
            start_date = end_date_obj - relativedelta(weeks=interval_count)
        elif period == 'month':
            start_date = end_date_obj - relativedelta(months=interval_count)
        elif period == 'quarter':
            start_date = end_date_obj - relativedelta(months=3 * interval_count)
        elif period == 'day':
            start_date = end_date_obj - relativedelta(days=interval_count)
        
        return start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_period(
    period_name: Annotated[str, Field(description="The name of the period ('January'-'December', 'Q1'-'Q4', 'Spring', 'Summer', 'Fall', 'Winter')")],
    year: Annotated[int, Field(description="The year in which the period falls")]
) -> Union[Dict[str, str], str]:
    """Calculate the start and end date based on the period name and year."""
    try:
        if not isinstance(period_name, str):
            return "Error: period_name must be a string"
        if not isinstance(year, int):
            return "Error: year must be an integer"

        valid_periods = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Q1', 'Q2', 'Q3', 'Q4', 'Spring', 'Summer', 'Fall', 'Winter']
        if period_name not in valid_periods:
            return f"Error: Invalid period_name. Must be one of {', '.join(valid_periods)}"

        if year < 1 or year > 9999:
            return "Error: year must be between 1 and 9999"

        if 'Q' in period_name:
            quarter = int(period_name[1])
            start_date = datetime(year, 3 * quarter - 2, 1)
            end_date = start_date + relativedelta(months=3) 
        elif period_name in ['Spring', 'Summer', 'Fall', 'Winter']:
            seasons = {'Spring': 3, 'Summer': 6, 'Fall': 9, 'Winter': 12}
            start_date = datetime(year, seasons[period_name], 1)
            end_date = start_date + relativedelta(months=3) 
        else:  # Month
            try:
                month = datetime.strptime(period_name, '%B').month
                start_date = datetime(year, month, 1)
                end_date = start_date + relativedelta(months=1) 
            except ValueError:
                return f"Error: Invalid month name '{period_name}'"

        return {
            "start_date": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'), 
            "end_date": end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        }
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_non_transferred_case_ids(
    start_date: Annotated[str, Field(description="Start date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    end_date: Annotated[str, Field(description="End date in format 'YYYY-MM-DDTHH:MM:SSZ'")]
) -> Union[List[str], str]:
    """Retrieve case IDs for cases that were not transferred between agents in the specified period."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(start_date, str) or not isinstance(end_date, str):
            return "Error: start_date and end_date must be strings"
        
        try:
            datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
            datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return "Error: start_date and end_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"

        query = f"""
            SELECT CaseId__c
            FROM CaseHistory__c
            WHERE Field__c = 'Owner Assignment'
            AND CreatedDate >= {start_date} AND CreatedDate <= {end_date}
            GROUP BY CaseId__c
            HAVING COUNT(Id) = 1
        """
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        return [record['CaseId__c'] for record in result]
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_agent_handled_cases_by_period(
    start_date: Annotated[str, Field(description="Start date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    end_date: Annotated[str, Field(description="End date in format 'YYYY-MM-DDTHH:MM:SSZ'")]
) -> Union[Dict[str, int], str]:
    """Retrieve the number of cases handled by each agent within the specified period."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(start_date, str) or not isinstance(end_date, str):
            return "Error: start_date and end_date must be strings"
        
        try:
            datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
            datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return "Error: start_date and end_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"

        query = f"""
            SELECT NewValue__c, COUNT(Id) CaseCount
            FROM CaseHistory__c
            WHERE Field__c = 'Owner Assignment'
            AND CreatedDate >= {start_date} AND CreatedDate <= {end_date}
            GROUP BY NewValue__c
        """
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        return {record['NewValue__c']: record['CaseCount'] for record in result}
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_qualified_agent_ids_by_case_count(
    agent_handled_cases: Annotated[Dict[str, int], Field(description="Dictionary where keys are agent IDs and values are the number of cases handled by each agent")],
    n_cases: Annotated[int, Field(description="The minimum number of cases an agent must have handled to be included")]
) -> Union[List[str], str]:
    """Filters agent IDs based on the number of cases they have handled. """
    try:
        # Check if agent_handled_cases is a dictionary
        if not isinstance(agent_handled_cases, dict):
            return "Error: agent_handled_cases must be a dictionary"

        # Check if n_cases is an integer
        if not isinstance(n_cases, int):
            return "Error: n_cases must be an integer"

        # Filter agent IDs based on case count
        qualified_agents = [agent_id for agent_id, count in agent_handled_cases.items() if count > n_cases]

        return qualified_agents

    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_agent_transferred_cases_by_period(
    start_date: Annotated[str, Field(description="Start date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    end_date: Annotated[str, Field(description="End date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    qualified_agent_ids: Annotated[Optional[List[str]], Field(description="Optional list of agent IDs to filter by")] = None
) -> Union[Dict[str, int], str]:
    """Retrieve the number of cases transferred between agents within the specified period."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(start_date, str) or not isinstance(end_date, str):
            return "Error: start_date and end_date must be strings"
        
        if qualified_agent_ids is not None and not isinstance(qualified_agent_ids, list):
            return "Error: qualified_agent_ids must be a list"
        
        try:
            datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%SZ')
            datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return "Error: start_date and end_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"

        query = f"""
            SELECT OldValue__c, COUNT(Id) TransferCount
            FROM CaseHistory__c
            WHERE Field__c = 'Owner Assignment'
            AND OldValue__c != NULL
            AND CreatedDate >= {start_date} AND CreatedDate <= {end_date}
        """
        
        if qualified_agent_ids:
            if len(qualified_agent_ids) == 1:
                query += f" AND OldValue__c = '{qualified_agent_ids[0]}'"
            else:
                query += f" AND OldValue__c IN {tuple(qualified_agent_ids)}"
                
        query += " GROUP BY OldValue__c"
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        return {record['OldValue__c']: record['TransferCount'] for record in result}
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_shipping_state(
    cases: Annotated[List[Dict], Field(description="List of cases, each containing an AccountId")]
) -> Union[List[Dict], str]:
    """Add shipping state information to the provided cases based on their associated accounts."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(cases, list):
            return "Error: Input 'cases' must be a list"

        if not cases:
            return cases

        for case in cases:
            if not isinstance(case, dict):
                return "Error: Each case must be a dictionary"
            if 'AccountId' not in case:
                return "Error: Each case dictionary must contain an 'AccountId' key"

        account_ids = [case['AccountId'] for case in cases]
        query = """
            SELECT Id, ShippingState
            FROM Account
        """
        if len(account_ids) == 1:
            query += f" WHERE Id = '{account_ids[0]}'"
        else:
            query += f" WHERE Id IN {tuple(account_ids)}"
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        account_states = {record['Id']: record['ShippingState'] for record in result}
        for case in cases:
            case['ShippingState'] = account_states.get(case['AccountId'])
        
        return cases
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def calculate_average_handle_time(
    cases: Annotated[List[Dict], Field(description="List of cases with CreatedDate, ClosedDate, and OwnerId")]
) -> Union[Dict[str, float], str]:
    """
    Calculate the average handle time for each agent based on a list of cases.

    This function computes the average time taken by each agent to handle their assigned cases.
    The handle time for a case is calculated as the difference between its closed date and created date.
    """
    try:
        if not isinstance(cases, list):
            return "Error: Input 'cases' must be a list"

        agent_handle_times = defaultdict(list)
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                return f"Error: Item at index {index} in cases is not a dictionary"
            
            required_keys = ['CreatedDate', 'ClosedDate', 'OwnerId']
            for key in required_keys:
                if key not in case:
                    return f"Error: '{key}' not found in case record at index {index}"
            
            try:
                created_date = datetime.strptime(case['CreatedDate'], '%Y-%m-%dT%H:%M:%S.%f%z')
                closed_date = datetime.strptime(case['ClosedDate'], '%Y-%m-%dT%H:%M:%S.%f%z')
            except ValueError:
                return f"Error: Invalid date format at index {index}. Expected format: '%Y-%m-%dT%H:%M:%S.%f%z'"
            
            if closed_date < created_date:
                return f"Error: ClosedDate is earlier than CreatedDate at index {index}"
            
            handle_time = (closed_date - created_date).total_seconds() / 60
            agent_handle_times[case['OwnerId']].append(handle_time)
        
        result = {}
        for agent, times in agent_handle_times.items():
            if times:
                result[agent] = sum(times) / len(times)
        
        return result
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def calculate_region_average_closure_times(
    cases: Annotated[List[Dict], Field(description="List of cases with ShippingState, CreatedDate, and ClosedDate")]
) -> Union[Dict[str, float], str]:
    """Calculate average case closure times grouped by region (shipping state)."""
    try:
        if not isinstance(cases, list):
            return "Error: Input 'cases' must be a list"

        if not cases:
            return "Error: Input 'cases' is empty"

        region_closure_times = defaultdict(list)
        for case in cases:
            if not isinstance(case, dict):
                return "Error: Each case must be a dictionary"
            
            if 'ShippingState' not in case or 'CreatedDate' not in case or 'ClosedDate' not in case:
                return "Error: Each case must contain 'ShippingState', 'CreatedDate', and 'ClosedDate' keys"
            
            state = case['ShippingState']
            if state:
                try:
                    created_date = datetime.strptime(case['CreatedDate'], '%Y-%m-%dT%H:%M:%S.%f%z')
                    closed_date = datetime.strptime(case['ClosedDate'], '%Y-%m-%dT%H:%M:%S.%f%z')
                except ValueError:
                    return "Error: Invalid date format. Expected format: 'YYYY-MM-DDTHH:MM:SS.000Z'"
                
                closure_time = (closed_date - created_date).total_seconds()
                region_closure_times[state].append(closure_time)

        return {region: sum(times) / len(times) for region, times in region_closure_times.items()}
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_account_id_by_contact_id(
    contact_id: Annotated[str, Field(description="The ID of the contact")]
) -> Union[str, None, str]:
    """Retrieve the Account ID associated with a given Contact ID."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(contact_id, str):
            return "Error: contact_id must be a string"

        if not contact_id:
            return "Error: contact_id cannot be empty"

        query = f"""
            SELECT AccountId
            FROM Contact
            WHERE Id = '{contact_id}'
            LIMIT 1
        """
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        if len(result) > 0:
            return result[0]['AccountId']
        return None
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_purchase_history(
    account_id: Annotated[str, Field(description="The ID of the account")],
    purchase_date: Annotated[str, Field(description="The date of purchase in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    related_product_ids: Annotated[List[str], Field(description="List of product IDs to search for")]
) -> Union[List[Dict], str]:
    """Retrieve purchase history for a specific account, date, and set of products."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(account_id, str):
            return "Error: account_id must be a string"
        if not isinstance(purchase_date, str):
            return "Error: purchase_date must be a string"
        if not isinstance(related_product_ids, list):
            return "Error: related_product_ids must be a list"

        try:
            datetime.strptime(purchase_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            return "Error: purchase_date must be in format 'YYYY-MM-DDTHH:MM:SSZ'"

        query = f"""
            SELECT Product2Id
            FROM OrderItem
            WHERE OrderItem.Order.AccountId = '{account_id}'
            AND OrderItem.Order.EffectiveDate = {purchase_date.split('T')[0]}
            AND Product2Id IN ('{"','".join(related_product_ids)}')
            AND OrderItem.Order.Status = 'Activated'
        """
        
        result, _ = sf_connector.run_query(query)
        return result
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_month_to_case_count(
    cases: Annotated[List[Dict], Field(description="List of cases with CreatedDate")]
) -> Union[Dict[str, int], str]:
    """Count the number of cases created in each month."""
    try:
        if not isinstance(cases, list):
            return "Error: Input must be a list of dictionaries"

        case_counts = defaultdict(int)
        for case in cases:
            if not isinstance(case, dict):
                return "Error: Each case must be a dictionary"
            if 'CreatedDate' not in case:
                return "Error: Each case must have a 'CreatedDate' key"
            
            try:
                case_date = datetime.strptime(case['CreatedDate'], '%Y-%m-%dT%H:%M:%S.%f%z')
            except ValueError:
                return "Error: Invalid date format. Expected format: 'YYYY-MM-DDTHH:MM:SS.000Z'"
            
            month_key = case_date.strftime('%B')
            case_counts[month_key] += 1
        
        return dict(case_counts)
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def search_products(
    search_term: Annotated[str, Field(description="Term to search for in product names and descriptions")]
) -> Union[List[Dict], str]:
    """Search for products based on a given search term."""
    if not sf_connector:
        return "Error: Salesforce connection not configured"

    if not isinstance(search_term, str):
        return "Error: search_term must be a string"

    if not search_term.strip():
        return "Error: search_term cannot be empty"

    sosl_query = f"""
        FIND {{{search_term}}} IN ALL FIELDS 
        RETURNING Product2(Id, Name, Description)
    """.strip()

    result, status = sf_connector.run_query(sosl_query)
    if status == 0:
        return result

    return result

@mcp.tool
def find_id_with_max_value(
    values_by_id: Annotated[Dict[str, Union[int, float]], Field(description="Dictionary with IDs as keys and their corresponding numeric values")]
) -> Union[List[str], str]:
    """Identifies the IDs with the maximum value from a dictionary."""
    try:
        if not isinstance(values_by_id, dict):
            return "Error: Input must be a dictionary"
        
        if not values_by_id:
            return []
        
        if not all(isinstance(value, (int, float)) for value in values_by_id.values()):
            return "Error: All values in the dictionary must be numeric"
        
        max_value = max(values_by_id.values())
        return [key for key, value in values_by_id.items() if value == max_value]
    
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def find_id_with_min_value(
    values_by_id: Annotated[Dict[str, Union[int, float]], Field(description="Dictionary with IDs as keys and their corresponding numeric values")]
) -> Union[List[str], str]:
    """Identifies the IDs with the minimum value from a dictionary."""
    try:
        if not isinstance(values_by_id, dict):
            return "Error: Input must be a dictionary"
        
        if not values_by_id:
            return []
        
        if not all(isinstance(value, (int, float)) for value in values_by_id.values()):
            return "Error: All values in the dictionary must be numeric"
        
        min_value = min(values_by_id.values())
        return [key for key, value in values_by_id.items() if value == min_value]
    
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_agents_with_max_cases(
    subset_cases: Annotated[List[Dict], Field(description="List of case records with OwnerId")]
) -> Union[List[str], str]:
    """Returns a list of agent IDs with the maximum number of cases from the given subset of cases."""
    try:
        if not isinstance(subset_cases, list):
            return "Error: Input 'subset_cases' must be a list"

        agent_issue_counts = {}
        for index, record in enumerate(subset_cases):
            if not isinstance(record, dict):
                return f"Error: Item at index {index} in subset_cases is not a dictionary"
            
            if 'OwnerId' not in record:
                return f"Error: 'OwnerId' not found in case record at index {index}"
            
            agent_id = record['OwnerId']
            if not isinstance(agent_id, str):
                return f"Error: 'OwnerId' at index {index} is not a string"
            
            agent_issue_counts[agent_id] = agent_issue_counts.get(agent_id, 0) + 1

        if agent_issue_counts:
            max_count = max(agent_issue_counts.values())
            return [agent for agent, count in agent_issue_counts.items() if count == max_count]
        else:
            return []
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_agents_with_min_cases(
    subset_cases: Annotated[List[Dict], Field(description="List of case records with OwnerId")]
) -> Union[List[str], str]:
    """Returns a list of agent IDs with the minimum number of cases from the given subset of cases."""
    try:
        if not isinstance(subset_cases, list):
            return "Error: Input 'subset_cases' must be a list"

        agent_issue_counts = {}
        for index, record in enumerate(subset_cases):
            if not isinstance(record, dict):
                return f"Error: Item at index {index} in subset_cases is not a dictionary"
            
            if 'OwnerId' not in record:
                return f"Error: 'OwnerId' not found in case record at index {index}"
            
            agent_id = record['OwnerId']
            if not isinstance(agent_id, str):
                return f"Error: 'OwnerId' at index {index} is not a string"
            
            agent_issue_counts[agent_id] = agent_issue_counts.get(agent_id, 0) + 1

        if agent_issue_counts:
            min_count = min(agent_issue_counts.values())
            return [agent for agent, count in agent_issue_counts.items() if count == min_count]
        else:
            return []
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

# TODO: Only used in CRMArena, not CRMArenaPro
# @mcp.tool
# def get_email_messages_by_case_id(
#     case_id: Annotated[str, Field(description="The ID of the case to retrieve email messages for")]
# ) -> Union[List[Dict], str]:
#     """Retrieves email messages associated with a specific case ID."""
#     try:        
#         if not sf_connector:
#             return "Error: Salesforce connection not configured"

#         if not isinstance(case_id, str):
#             return "Error: case_id must be a string"

#         query = f"""
#             SELECT Subject, TextBody, FromAddress, ToAddress, MessageDate
#             FROM EmailMessage
#             WHERE ParentId = '{case_id}'
#         """
#         result, status = sf_connector.run_query(query)
#         return result
#     except Exception as e:
#         return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_livechat_transcript_by_case_id(
    case_id: Annotated[str, Field(description="The ID of the case to retrieve live chat transcripts for")]
) -> Union[List[Dict], str]:
    """Retrieves live chat transcripts associated with a specific case ID."""
    try:        
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(case_id, str):
            return "Error: case_id must be a string"

        query = f"""
            SELECT Body, EndTime
            FROM LiveChatTranscript
            WHERE CaseId = '{case_id}'
        """
        result, status = sf_connector.run_query(query)
        return result
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_issue_counts(
    start_date: Annotated[str, Field(description="Start date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    end_date: Annotated[str, Field(description="End date in format 'YYYY-MM-DDTHH:MM:SSZ'")],
    order_item_ids: Annotated[List[str], Field(description="List of order item IDs to filter issues by")]
) -> Union[Dict[str, int], str]:
    """Retrieves the issue counts for products within a given time period."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(start_date, str) or not isinstance(end_date, str):
            return "Error: start_date and end_date must be strings"
        if not isinstance(order_item_ids, list) or not order_item_ids:
            return "Error: order_item_ids must be a non-empty list"

        query = f"""
            SELECT IssueId__c, COUNT(Id) IssueCount
            FROM Case
            WHERE OrderItemId__c IN ('{"','".join(map(str, order_item_ids))}')
            AND CreatedDate >= {start_date}
            AND CreatedDate <= {end_date}
            GROUP BY IssueId__c
            ORDER BY COUNT(Id) DESC
        """
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        return {record["IssueId__c"]: record["IssueCount"] for record in result}
    
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_issues() -> Union[List[Dict], str]:
    """Retrieves a list of issue records from Salesforce."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        query = """
            SELECT Id, Name
            FROM Issue__c
        """
        result, status = sf_connector.run_query(query)
        return result if status == 1 else result
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def get_order_item_ids_by_product(
    product_id: Annotated[str, Field(description="The ID of the product")]
) -> Union[List[str], str]:
    """Retrieves the order item IDs associated with a given product."""
    try:
        if not sf_connector:
            return "Error: Salesforce connection not configured"

        if not isinstance(product_id, str):
            return "Error: product_id must be a string"

        if not product_id:
            return "Error: product_id cannot be empty"

        query = f"""
            SELECT Id
            FROM OrderItem
            WHERE Product2Id = '{product_id}'
        """
        
        result, status = sf_connector.run_query(query)
        if status == 0:
            return result
        
        return [record["Id"] for record in result]
    
    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

@mcp.tool
def issue_soql_query(
    query: Annotated[str, Field(description="The SOQL query string to execute")]
) -> Union[List[Dict], str]:
    """Executes a SOQL (Salesforce Object Query Language) query to retrieve data from Salesforce."""
    if not sf_connector:
        return "Error: Salesforce connection not configured"
    result, _ = sf_connector.run_query(query)
    return result

@mcp.tool
def issue_sosl_query(
    query: Annotated[str, Field(description="The SOSL query string to execute")]
) -> Union[List[Dict], str]:
    """Executes a SOSL (Salesforce Object Search Language) query to retrieve data from Salesforce."""
    if not sf_connector:
        return "Error: Salesforce connection not configured"
    result, _ = sf_connector.run_query(query)
    return result

if __name__ == "__main__":
    # Set up Salesforce connection when server starts
    try:
        import os
        config = {
            "username": os.environ["SALESFORCE_USERNAME"],
            "password": os.environ["SALESFORCE_PASSWORD"],
            "security_token": os.environ["SALESFORCE_SECURITY_TOKEN"]
        }
        set_salesforce_connector(config=config)
        logger.info("Successfully connected to Salesforce")
    except KeyError as e:
        logger.error(f"Missing required environment variable: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to connect to Salesforce: {str(e)}")
        raise

    mcp.run()
