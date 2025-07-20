import pytest
from typing import Dict, List, Union, cast
from fastmcp import Client
from pytest import mark
import pytest_asyncio
from datetime import datetime

from custom_mcp.salesforce_mcp import (
    mcp,
    set_salesforce_connector,
)

# Type aliases
SalesforceRecord = Dict[str, Union[str, None]]
QueryResult = Union[List[SalesforceRecord], str]

def get_str_value(record: SalesforceRecord, key: str, default: str = '') -> str:
    """Helper function to get string value from record with type checking."""
    value = record.get(key)
    return value if isinstance(value, str) else default

def verify_query_result(result: QueryResult) -> List[SalesforceRecord]:
    """Helper function to verify and cast query result."""
    assert result is not None, "Query result should not be None"
    assert isinstance(result, list), "Query result should be a list"
    return cast(List[SalesforceRecord], result)

@pytest_asyncio.fixture(scope="module")
async def salesforce_client():
    """Setup Salesforce connection and return MCP server for testing."""
    config = {
        "username": "crmarena_b2b@gmaill.com",
        "password": "crmarenatest",
        "security_token": "zdaqqSYBEQTjjLuq0zLUHkC3"
    }
    
    # Initialize and set the global connector
    set_salesforce_connector(config=config)
    
    # Return the MCP server
    return mcp

@mark.asyncio
async def test_soql_query(salesforce_client) -> None:
    """Test SOQL query execution with a simple Account query."""
    query = """SELECT Id, Name FROM Account LIMIT 5"""
    
    # Execute query through MCP client
    async with Client(salesforce_client) as client:
        result = await client.call_tool("issue_soql_query", {"query": query})
        assert result.structured_content is not None, "Query result should not be None"
        # check that structured_content has the key 'result'
        assert 'result' in result.structured_content, "Query result should contain 'result'' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        assert len(result_data) > 0, "Should return at least one record"
    
        # Record format validations
        assert all(isinstance(r, dict) for r in result_data), "All records should be dictionaries"
        assert all(all(key in r for key in ['Id', 'Name']) for r in result_data), "All records should have Id and Name"
        
        # Salesforce-specific validations
        assert all(get_str_value(r, 'Id').startswith('001') for r in result_data), "Account IDs should start with '001'"
        assert all(get_str_value(r, 'Name') for r in result_data), "All records should have non-empty Names"

@mark.asyncio
async def test_sosl_query(salesforce_client) -> None:
    """Test SOSL query execution with a simple search query."""
    query = """FIND {Tech} IN ALL FIELDS 
        RETURNING Account(Id, Name WHERE Name LIKE '%Tech%')"""
    
    # Execute search through MCP client
    async with Client(salesforce_client) as client:
        result = await client.call_tool("issue_sosl_query", {"query": query})
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        assert len(result_data) > 0, "Should return at least one record with 'Tech' in name"
    
        # Record format validations
        assert all(isinstance(r, dict) for r in result_data), "All records should be dictionaries"
        assert all(all(key in r for key in ['Id', 'Name']) for r in result_data), "All records should have Id and Name"
        
        # Content validations
        assert all('Tech' in get_str_value(r, 'Name') for r in result_data), "All records should contain 'Tech' in name"
        assert all(get_str_value(r, 'Id').startswith('001') for r in result_data), "Account IDs should start with '001'"

# @mark.asyncio
# async def test_get_email_messages_by_case_id(salesforce_client) -> None:
#     """Test email message retrieval with field validation."""
#     case_id = "500Wt00000DDNYoIAP"  # Using known case ID from sample data

#     async with Client(salesforce_client) as client:
#         result = await client.call_tool("get_email_messages_by_case_id", {"case_id": case_id})
#         assert result.structured_content is not None, "Query result should not be None"
#         assert 'result' in result.structured_content, "Query result should contain 'result' key"
#         assert isinstance(result.structured_content['result'], list), "Query result should be a list"
#         result_data = verify_query_result(result.structured_content['result'])
        
#         assert len(result_data) > 0, "Should return at least one email message"
#         # Field presence validation
#         required_fields = ['Subject', 'TextBody', 'FromAddress', 'ToAddress', 'MessageDate']
#         for email in result_data:
#             assert all(field in email for field in required_fields), \
#                 f"Email should contain all fields: {required_fields}"
            
#             # Validate email format
#             assert '@' in get_str_value(email, 'FromAddress'), "FromAddress should be valid email"
#             assert '@' in get_str_value(email, 'ToAddress'), "ToAddress should be valid email"
            
#             # Validate date format
#             message_date = get_str_value(email, 'MessageDate')
#             assert message_date, "MessageDate should not be empty"
#             try:
#                 datetime.strptime(message_date, '%Y-%m-%dT%H:%M:%S.%f%z')
#             except ValueError:
#                 pytest.fail("MessageDate should be in format 'YYYY-MM-DDTHH:MM:SS.000+0000'")

#         # Test non-existent case ID
#         invalid_result = await client.call_tool("get_email_messages_by_case_id", {
#             "case_id": "500000000000000000"
#         })
#         assert invalid_result.structured_content is not None
#         assert 'result' in invalid_result.structured_content
#         assert len(invalid_result.structured_content['result']) == 0

@mark.asyncio
async def test_get_livechat_transcript_by_case_id(salesforce_client) -> None:
    """Test live chat transcript retrieval with content validation."""
    case_id = "500Wt00000DDQRsIAP"  # Using known case ID from sample data

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_livechat_transcript_by_case_id", {"case_id": case_id})
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        
        assert len(result_data) > 0, "Should return at least one transcript"
        for transcript in result_data:
            # Verify required fields
            assert 'Body' in transcript, "Transcript should contain Body"
            assert 'EndTime' in transcript, "Transcript should contain EndTime"
            
            # Validate chat log format
            body = get_str_value(transcript, 'Body')
            assert '[' in body and ']' in body, "Chat log should contain timestamped entries"
            
            # Validate date format
            end_time = get_str_value(transcript, 'EndTime')
            assert end_time, "EndTime should not be empty"
            try:
                datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%f%z')
            except ValueError:
                pytest.fail("EndTime should be in format 'YYYY-MM-DDTHH:MM:SS.000+0000'")

        # Test non-existent case ID
        invalid_result = await client.call_tool("get_livechat_transcript_by_case_id", {
            "case_id": "500000000000000000"
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert len(invalid_result.structured_content['result']) == 0

@mark.asyncio
async def test_get_cases(salesforce_client) -> None:
    """Test get_cases with various filtering options."""
    # Test date range filtering
    start_date = "2023-09-30T00:00:00Z"
    end_date = "2023-10-03T00:00:00Z"
    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_cases", {
            "start_date": start_date,
            "end_date": end_date
        })
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        assert len(result_data) > 0, "Should return at least one case"
        
        # Verify cases are within date range
        for case in result_data:
            case_date = get_str_value(case, 'CreatedDate')
            assert case_date, "CreatedDate should not be empty"
            assert start_date <= case_date <= end_date, "Case date should be within specified range"
    
    # Test agent filtering
    agent_id = "005Wt000003NIc3IAG"  # Known agent ID from sample data
    async with Client(salesforce_client) as client:
        agent_cases = await client.call_tool("get_cases", {
            "agent_ids": [agent_id]
        })
        assert agent_cases.structured_content is not None, "Query result should not be None"
        assert 'result' in agent_cases.structured_content, "Query result should contain 'result' key"
        assert isinstance(agent_cases.structured_content['result'], list), "Query result should be a list"
        agent_cases_data = verify_query_result(agent_cases.structured_content['result'])
        assert len(agent_cases_data) > 0, "Should return at least one case"
        assert all(get_str_value(case, 'OwnerId') == agent_id for case in agent_cases_data), "All cases should belong to specified agent"
    
    # Test case ID filtering
    case_id = "500Wt00000DDNYoIAP"  # Known case ID from sample data
    async with Client(salesforce_client) as client:
        specific_case = await client.call_tool("get_cases", {
            "case_ids": [case_id]
        })
        assert specific_case.structured_content is not None, "Query result should not be None"
        assert 'result' in specific_case.structured_content, "Query result should contain 'result' key"
        assert isinstance(specific_case.structured_content['result'], list), "Query result should be a list"
        specific_case_data = verify_query_result(specific_case.structured_content['result'])
        assert len(specific_case_data) == 1, "Should return exactly one case"

    # Test status filtering
    async with Client(salesforce_client) as client:
        status_cases = await client.call_tool("get_cases", {
            "statuses": ["Closed"]
        })
        assert status_cases.structured_content is not None, "Query result should not be None"
        assert 'result' in status_cases.structured_content, "Query result should contain 'result' key"
        assert isinstance(status_cases.structured_content['result'], list), "Query result should be a list"
        status_cases_data = verify_query_result(status_cases.structured_content['result'])
        assert len(status_cases_data) > 0, "Should return at least one case"

@mark.asyncio
async def test_get_issues(salesforce_client) -> None:
    """Test issue record retrieval."""
    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_issues", {})
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        assert len(result_data) > 0, "Should return at least one issue"
        
        # Field validation
        for issue in result_data:
            assert isinstance(issue, dict), "Each issue should be a dictionary"
            assert all(key in issue for key in ['Id', 'Name']), "Issues should have Id and Name"
            assert get_str_value(issue, 'Id').startswith('a03'), "Issue IDs should start with 'a03'"
            assert get_str_value(issue, 'Name'), "Issue Name should not be empty"

@mark.asyncio
async def test_get_purchase_history(salesforce_client) -> None:
    """Test purchase history retrieval with product filtering."""
    account_id = "001Wt00000PFttwIAD"  # Known account ID from sample data
    purchase_date = "2023-06-25T00:00:00Z"
    product_id = "01tWt000006hTUkIAM"  # Known product ID from sample data

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_purchase_history", {
            "account_id": account_id,
            "purchase_date": purchase_date,
            "related_product_ids": [product_id]
        })
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        
        assert len(result_data) > 0, "Should return at least one order item"
        for order_item in result_data:
            # Verify required fields
            assert isinstance(order_item, dict), "Each order item should be a dictionary"
            assert 'Product2Id' in order_item, "Order item should contain Product2Id"
            
            # Validate Product ID format
            assert get_str_value(order_item, 'Product2Id').startswith('01t'), \
                "Product IDs should start with '01t'"

        # Test non-existent account
        invalid_result = await client.call_tool("get_purchase_history", {
            "account_id": "001000000000000000",
            "purchase_date": purchase_date,
            "related_product_ids": [product_id]
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert len(invalid_result.structured_content['result']) == 0

@mark.asyncio
async def test_get_period(salesforce_client) -> None:
    """Test period date range calculation."""
    async with Client(salesforce_client) as client:
        # Test month period
        month_result = await client.call_tool("get_period", {
            "period_name": "March",
            "year": 2023
        })
        assert month_result.structured_content is not None, "Result should not be None"
        assert 'result' in month_result.structured_content, "Result should contain 'result' key"
        assert month_result.structured_content['result'] is not None, "Result value should not be None"
        month_dates = month_result.structured_content['result']
        assert isinstance(month_dates, dict), "Result should be a dictionary"
        assert all(key in month_dates for key in ['start_date', 'end_date']), \
            "Result should contain start_date and end_date"
        assert month_dates['start_date'] == '2023-03-01T00:00:00Z'
        assert month_dates['end_date'] == '2023-04-01T00:00:00Z'

        # Test quarter period
        quarter_result = await client.call_tool("get_period", {
            "period_name": "Q2",
            "year": 2023
        })
        assert quarter_result.structured_content is not None
        assert 'result' in quarter_result.structured_content
        assert quarter_result.structured_content['result'] is not None
        quarter_dates = quarter_result.structured_content['result']
        assert quarter_dates['start_date'] == '2023-04-01T00:00:00Z'
        assert quarter_dates['end_date'] == '2023-07-01T00:00:00Z'

        # Test season period
        season_result = await client.call_tool("get_period", {
            "period_name": "Summer",
            "year": 2023
        })
        assert season_result.structured_content is not None
        assert 'result' in season_result.structured_content
        assert season_result.structured_content['result'] is not None
        season_dates = season_result.structured_content['result']
        assert season_dates['start_date'] == '2023-06-01T00:00:00Z'
        assert season_dates['end_date'] == '2023-09-01T00:00:00Z'

@mark.asyncio
async def test_get_start_date(salesforce_client) -> None:
    """Test start date calculation from end date and interval."""
    end_date = "2023-12-31T00:00:00Z"

    async with Client(salesforce_client) as client:
        # Test day interval
        day_result = await client.call_tool("get_start_date", {
            "end_date": end_date,
            "period": "day",
            "interval_count": 7
        })
        assert day_result.structured_content is not None
        assert 'result' in day_result.structured_content
        assert day_result.structured_content['result'] == '2023-12-24T00:00:00Z'

        # Test week interval
        week_result = await client.call_tool("get_start_date", {
            "end_date": end_date,
            "period": "week",
            "interval_count": 2
        })
        assert week_result.structured_content is not None
        assert 'result' in week_result.structured_content
        assert week_result.structured_content['result'] == '2023-12-17T00:00:00Z'

        # Test quarter interval
        quarter_result = await client.call_tool("get_start_date", {
            "end_date": end_date,
            "period": "quarter",
            "interval_count": 1
        })
        assert quarter_result.structured_content is not None
        assert 'result' in quarter_result.structured_content
        assert quarter_result.structured_content['result'] == '2023-09-30T00:00:00Z'

@mark.asyncio
async def test_get_non_transferred_case_ids(salesforce_client) -> None:
    """Test retrieval of non-transferred case IDs."""
    start_date = "2023-09-30T00:00:00Z"
    end_date = "2023-10-03T00:00:00Z"

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_non_transferred_case_ids", {
            "start_date": start_date,
            "end_date": end_date
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        case_ids = result.structured_content['result']
        for case_id in case_ids:
            assert isinstance(case_id, str), "Case IDs should be strings"

@mark.asyncio
async def test_get_agent_handled_cases_by_period(salesforce_client) -> None:
    """Test agent case count retrieval by period."""
    start_date = "2023-09-30T00:00:00Z"
    end_date = "2023-10-03T00:00:00Z"

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_agent_handled_cases_by_period", {
            "start_date": start_date,
            "end_date": end_date
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], dict), "Result should be a dictionary"
        case_counts = result.structured_content['result']
        for agent_id, count in case_counts.items():
            assert isinstance(agent_id, str), "Agent IDs should be strings"
            assert isinstance(count, int), "Case counts should be integers"

@mark.asyncio
async def test_calculate_average_handle_time(salesforce_client) -> None:
    """Test average handle time calculation for cases."""
    test_cases = [
        {
            'CreatedDate': '2023-10-01T10:00:00.000+0000',
            'ClosedDate': '2023-10-01T11:30:00.000+0000',
            'OwnerId': '005Wt000003NIc3IAG'
        },
        {
            'CreatedDate': '2023-10-01T14:00:00.000+0000',
            'ClosedDate': '2023-10-01T14:45:00.000+0000',
            'OwnerId': '005Wt000003NIc3IAG'
        }
    ]

    async with Client(salesforce_client) as client:
        result = await client.call_tool("calculate_average_handle_time", {
            "cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert result.structured_content['result'] is not None, "Result value should not be None"
        
        handle_times = result.structured_content['result']
        assert isinstance(handle_times, dict), "Result should be a dictionary"
        assert '005Wt000003NIc3IAG' in handle_times, "Result should contain agent ID"
        
        # First case took 90 minutes, second took 45 minutes, average should be 67.5
        assert abs(handle_times['005Wt000003NIc3IAG'] - 67.5) < 0.1, \
            "Average handle time calculation is incorrect"

        # Test invalid date format
        invalid_cases = [
            {
                'CreatedDate': 'invalid-date',
                'ClosedDate': '2023-10-01T11:30:00.000+0000',
                'OwnerId': '005Wt000003NIc3IAG'
            }
        ]
        invalid_result = await client.call_tool("calculate_average_handle_time", {
            "cases": invalid_cases
        })
        assert invalid_result.structured_content is not None, "Result should not be None"
        assert 'result' in invalid_result.structured_content, "Result should contain 'result' key"
        assert invalid_result.structured_content['result'] is not None, "Result value should not be None"
        assert isinstance(invalid_result.structured_content['result'], str), \
            "Should return error message for invalid date format"
        assert "Invalid date format" in invalid_result.structured_content['result']

@mark.asyncio
async def test_calculate_region_average_closure_times(salesforce_client) -> None:
    """Test regional average case closure time calculation."""
    test_cases = [
        {
            'CreatedDate': '2023-10-01T10:00:00.000+0000',
            'ClosedDate': '2023-10-01T11:30:00.000+0000',
            'ShippingState': 'CA'
        },
        {
            'CreatedDate': '2023-10-01T14:00:00.000+0000',
            'ClosedDate': '2023-10-01T14:45:00.000+0000',
            'ShippingState': 'CA'
        },
        {
            'CreatedDate': '2023-10-01T09:00:00.000+0000',
            'ClosedDate': '2023-10-01T12:00:00.000+0000',
            'ShippingState': 'NY'
        }
    ]

    async with Client(salesforce_client) as client:
        result = await client.call_tool("calculate_region_average_closure_times", {
            "cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert result.structured_content['result'] is not None, "Result value should not be None"
        
        closure_times = result.structured_content['result']
        assert isinstance(closure_times, dict), "Result should be a dictionary"
        assert 'CA' in closure_times and 'NY' in closure_times, \
            "Result should contain both regions"
        
        # CA: First case 90 min (5400s), second case 45 min (2700s), avg 67.5 min (4050s)
        # NY: One case 180 min (10800s)
        assert abs(closure_times['CA'] - 4050) < 1, "CA average closure time is incorrect"
        assert abs(closure_times['NY'] - 10800) < 1, "NY average closure time is incorrect"

        # Test invalid date format
        invalid_cases = [
            {
                'CreatedDate': '2023-10-01T10:00:00.000+0000',
                'ClosedDate': 'invalid-date',
                'ShippingState': 'CA'
            }
        ]
        invalid_result = await client.call_tool("calculate_region_average_closure_times", {
            "cases": invalid_cases
        })
        assert invalid_result.structured_content is not None, "Result should not be None"
        assert 'result' in invalid_result.structured_content, "Result should contain 'result' key"
        assert invalid_result.structured_content['result'] is not None, "Result value should not be None"
        assert isinstance(invalid_result.structured_content['result'], str), \
            "Should return error message for invalid date format"
        assert "Invalid date format" in invalid_result.structured_content['result']

@mark.asyncio
async def test_get_month_to_case_count(salesforce_client) -> None:
    """Test case counting by month."""
    test_cases = [
        {'CreatedDate': '2023-10-01T10:00:00.000+0000'},
        {'CreatedDate': '2023-10-15T14:00:00.000+0000'},
        {'CreatedDate': '2023-11-01T09:00:00.000+0000'}
    ]

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_month_to_case_count", {
            "cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert result.structured_content['result'] is not None, "Result value should not be None"
        
        case_counts = result.structured_content['result']
        assert isinstance(case_counts, dict), "Result should be a dictionary"
        assert case_counts['October'] == 2, "October should have 2 cases"
        assert case_counts['November'] == 1, "November should have 1 case"

@mark.asyncio
async def test_get_order_item_ids_by_product(salesforce_client) -> None:
    """Test retrieval of order item IDs for a product."""
    product_id = "01tWt000006hTUkIAM"  # Known product ID from sample data

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_order_item_ids_by_product", {
            "product_id": product_id
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        
        order_item_ids = result.structured_content['result']
        for order_item_id in order_item_ids:
            assert isinstance(order_item_id, str), "Order item IDs should be strings"
            assert order_item_id.startswith('802'), "Order item IDs should start with '802'"

@mark.asyncio
async def test_get_issue_counts(salesforce_client) -> None:
    """Test issue counting for products."""
    start_date = "2023-09-30T00:00:00Z"
    end_date = "2023-10-03T00:00:00Z"
    order_item_ids = ["802Wt000007906kIAA"]  # Known order item ID from sample data

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_issue_counts", {
            "start_date": start_date,
            "end_date": end_date,
            "order_item_ids": order_item_ids
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], dict), "Result should be a dictionary"
        
        issue_counts = result.structured_content['result']
        for issue_id, count in issue_counts.items():
            assert isinstance(issue_id, str), "Issue IDs should be strings"
            assert isinstance(count, int), "Counts should be integers"

@mark.asyncio
async def test_get_qualified_agent_ids_by_case_count(salesforce_client) -> None:
    """Test filtering agents by case count."""
    agent_cases = {
        "005Wt000003NIc3IAG": 10,
        "005Wt000003NIc4IAG": 5,
        "005Wt000003NIc5IAG": 15
    }

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_qualified_agent_ids_by_case_count", {
            "agent_handled_cases": agent_cases,
            "n_cases": 7
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        
        qualified_agents = result.structured_content['result']
        assert len(qualified_agents) == 2, "Should have 2 qualified agents"
        assert "005Wt000003NIc3IAG" in qualified_agents, "First agent should be qualified"
        assert "005Wt000003NIc5IAG" in qualified_agents, "Third agent should be qualified"

@mark.asyncio
async def test_get_account_id_by_contact_id(salesforce_client) -> None:
    """Test account ID retrieval from contact ID."""
    contact_id = "003Wt000004XnABIA0"  # Known contact ID from sample data

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_account_id_by_contact_id", {
            "contact_id": contact_id
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        
        account_id = result.structured_content['result']
        if account_id is not None:
            assert isinstance(account_id, str), "Account ID should be a string"
            assert account_id.startswith('001'), "Account ID should start with '001'"

        # Test non-existent contact
        invalid_result = await client.call_tool("get_account_id_by_contact_id", {
            "contact_id": "003000000000000000"
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert invalid_result.structured_content['result'] is None

@mark.asyncio
async def test_search_knowledge_articles(salesforce_client) -> None:
    """Test knowledge article search functionality."""
    # Search for a term we know exists in the sample data
    search_term = "PCB"
    async with Client(salesforce_client) as client:
        result = await client.call_tool("search_knowledge_articles", {
            "search_term": search_term
        })
        assert result.structured_content is not None, "Query result should not be None"
        assert 'result' in result.structured_content, "Query result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Query result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        assert len(result_data) > 0, "Should return at least one article"
        
        # Content structure validations
        for article in result_data:
            assert isinstance(article, dict), "Each article should be a dictionary"
            assert all(key in article for key in ['Id', 'Title', 'Content']), \
                "Articles should have Id, Title, and Content"
            assert get_str_value(article, 'Id').startswith('ka0'), "Article IDs should start with 'ka0'"
            
            # Verify search term appears in content
            title = get_str_value(article, 'Title')
            content = get_str_value(article, 'Content')
            content_match = (
                search_term.lower() in title.lower() or 
                search_term.lower() in content.lower()
            )
            assert content_match, f"Search term '{search_term}' should appear in article title or content"

@mark.asyncio
async def test_get_agent_transferred_cases_by_period(salesforce_client) -> None:
    """Test retrieval of cases transferred between agents."""
    start_date = "2023-09-30T00:00:00Z"
    end_date = "2023-10-03T00:00:00Z"
    qualified_agent_ids = ["005Wt000003NIc3IAG", "005Wt000003NJ6gIAG"]  # Known agent IDs from sample data

    async with Client(salesforce_client) as client:
        # Test with qualified agent IDs
        result = await client.call_tool("get_agent_transferred_cases_by_period", {
            "start_date": start_date,
            "end_date": end_date,
            "qualified_agent_ids": qualified_agent_ids
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], dict), "Result should be a dictionary"
        
        transfer_counts = result.structured_content['result']
        for agent_id, count in transfer_counts.items():
            assert isinstance(agent_id, str), "Agent IDs should be strings"
            assert isinstance(count, int), "Transfer counts should be integers"
            assert agent_id.startswith('005'), "Agent IDs should start with '005'"

        # Test without qualified agent IDs filter
        unfiltered_result = await client.call_tool("get_agent_transferred_cases_by_period", {
            "start_date": start_date,
            "end_date": end_date
        })
        assert unfiltered_result.structured_content is not None
        assert 'result' in unfiltered_result.structured_content
        assert isinstance(unfiltered_result.structured_content['result'], dict)

        # Test invalid date format
        invalid_result = await client.call_tool("get_agent_transferred_cases_by_period", {
            "start_date": "invalid-date",
            "end_date": end_date
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert isinstance(invalid_result.structured_content['result'], str)
        assert "must be in format 'YYYY-MM-DDTHH:MM:SSZ'" in invalid_result.structured_content['result']

@mark.asyncio
async def test_search_products(salesforce_client) -> None:
    """Test product search functionality."""
    # Test basic search with minimal info
    search_term = "Layout"  # Known to match "AutoLayout Master"
    async with Client(salesforce_client) as client:
        result = await client.call_tool("search_products", {
            "search_term": search_term
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        basic_results = verify_query_result(result.structured_content['result'])
        
        # Verify basic result structure
        for product in basic_results:
            assert isinstance(product, dict), "Each product should be a dictionary"
            assert all(key in product for key in ['Name']), \
                "Basic results should have Name"

        # Test empty search term
        empty_result = await client.call_tool("search_products", {
            "search_term": "",
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert isinstance(empty_result.structured_content['result'], str)
        assert "search_term cannot be empty" in empty_result.structured_content['result']

@mark.asyncio
async def test_find_id_with_max_value(salesforce_client) -> None:
    """Test finding IDs with maximum value."""
    # Test normal case with single max
    test_data = {
        "id1": 10,
        "id2": 5,
        "id3": 15
    }
    async with Client(salesforce_client) as client:
        result = await client.call_tool("find_id_with_max_value", {
            "values_by_id": test_data
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        assert result.structured_content['result'] == ["id3"], "Should find ID with max value"

        # Test case with multiple max values
        test_data_tie = {
            "id1": 15,
            "id2": 10,
            "id3": 15
        }
        tie_result = await client.call_tool("find_id_with_max_value", {
            "values_by_id": test_data_tie
        })
        assert tie_result.structured_content is not None
        assert 'result' in tie_result.structured_content
        assert isinstance(tie_result.structured_content['result'], list)
        assert sorted(tie_result.structured_content['result']) == ["id1", "id3"], \
            "Should find all IDs with max value"

        # Test empty dictionary
        empty_result = await client.call_tool("find_id_with_max_value", {
            "values_by_id": {}
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert empty_result.structured_content['result'] == [], "Should return empty list for empty input"


@mark.asyncio
async def test_find_id_with_min_value(salesforce_client) -> None:
    """Test finding IDs with minimum value."""
    # Test normal case with single min
    test_data = {
        "id1": 10,
        "id2": 5,
        "id3": 15
    }
    async with Client(salesforce_client) as client:
        result = await client.call_tool("find_id_with_min_value", {
            "values_by_id": test_data
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        assert result.structured_content['result'] == ["id2"], "Should find ID with min value"

        # Test case with multiple min values
        test_data_tie = {
            "id1": 5,
            "id2": 10,
            "id3": 5
        }
        tie_result = await client.call_tool("find_id_with_min_value", {
            "values_by_id": test_data_tie
        })
        assert tie_result.structured_content is not None
        assert 'result' in tie_result.structured_content
        assert isinstance(tie_result.structured_content['result'], list)
        assert sorted(tie_result.structured_content['result']) == ["id1", "id3"], \
            "Should find all IDs with min value"

        # Test empty dictionary
        empty_result = await client.call_tool("find_id_with_min_value", {
            "values_by_id": {}
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert empty_result.structured_content['result'] == [], "Should return empty list for empty input"


@mark.asyncio
async def test_get_agents_with_max_cases(salesforce_client) -> None:
    """Test finding agents with maximum number of cases."""
    # Test normal case with single max
    test_cases = [
        {"OwnerId": "agent1", "CaseId": "case1"},
        {"OwnerId": "agent1", "CaseId": "case2"},
        {"OwnerId": "agent2", "CaseId": "case3"},
        {"OwnerId": "agent3", "CaseId": "case4"},
        {"OwnerId": "agent3", "CaseId": "case5"}
    ]
    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_agents_with_max_cases", {
            "subset_cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        assert sorted(result.structured_content['result']) == ["agent1", "agent3"], \
            "Should find agents with max cases"

        # Test empty case list
        empty_result = await client.call_tool("get_agents_with_max_cases", {
            "subset_cases": []
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert empty_result.structured_content['result'] == [], "Should return empty list for empty input"

        # Test invalid case records
        invalid_result = await client.call_tool("get_agents_with_max_cases", {
            "subset_cases": [{"InvalidKey": "value"}]
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert isinstance(invalid_result.structured_content['result'], str)
        assert "'OwnerId' not found" in invalid_result.structured_content['result']

@mark.asyncio
async def test_get_agents_with_min_cases(salesforce_client) -> None:
    """Test finding agents with minimum number of cases."""
    # Test normal case with single min
    test_cases = [
        {"OwnerId": "agent1", "CaseId": "case1"},
        {"OwnerId": "agent1", "CaseId": "case2"},
        {"OwnerId": "agent2", "CaseId": "case3"},
        {"OwnerId": "agent3", "CaseId": "case4"},
        {"OwnerId": "agent3", "CaseId": "case5"}
    ]
    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_agents_with_min_cases", {
            "subset_cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        assert result.structured_content['result'] == ["agent2"], \
            "Should find agent with min cases"

        # Test empty case list
        empty_result = await client.call_tool("get_agents_with_min_cases", {
            "subset_cases": []
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert empty_result.structured_content['result'] == [], "Should return empty list for empty input"

        # Test invalid case records
        invalid_result = await client.call_tool("get_agents_with_min_cases", {
            "subset_cases": [{"InvalidKey": "value"}]
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert isinstance(invalid_result.structured_content['result'], str)
        assert "'OwnerId' not found" in invalid_result.structured_content['result']

@mark.asyncio
async def test_get_shipping_state(salesforce_client) -> None:
    """Test adding shipping state information to cases based on their associated accounts."""
    # Test cases with valid AccountIds
    test_cases = [
        {"AccountId": "001Wt00000PFj4zIAD", "CaseNumber": "00001"},  # Known account with shipping state
        {"AccountId": "001000000000000000", "CaseNumber": "00002"}   # Non-existent account
    ]

    async with Client(salesforce_client) as client:
        result = await client.call_tool("get_shipping_state", {
            "cases": test_cases
        })
        assert result.structured_content is not None, "Result should not be None"
        assert 'result' in result.structured_content, "Result should contain 'result' key"
        assert isinstance(result.structured_content['result'], list), "Result should be a list"
        result_data = verify_query_result(result.structured_content['result'])
        
        # Verify shipping state was added to cases
        assert len(result_data) == len(test_cases), "Should return same number of cases"
        assert all('ShippingState' in case for case in result_data), \
            "All cases should have ShippingState field added"
        
        # First case should have a valid shipping state
        assert result_data[0]['ShippingState'] is not None, \
            "Known account should have shipping state"
        
        # Second case should have null shipping state
        assert result_data[1]['ShippingState'] is None, \
            "Non-existent account should have null shipping state"

        # Test empty case list
        empty_result = await client.call_tool("get_shipping_state", {
            "cases": []
        })
        assert empty_result.structured_content is not None
        assert 'result' in empty_result.structured_content
        assert empty_result.structured_content['result'] == [], \
            "Should return empty list for empty input"

        # Test case without AccountId
        invalid_result = await client.call_tool("get_shipping_state", {
            "cases": [{"CaseNumber": "00003"}]
        })
        assert invalid_result.structured_content is not None
        assert 'result' in invalid_result.structured_content
        assert isinstance(invalid_result.structured_content['result'], str)
        assert "Error: Each case dictionary must contain an 'AccountId' key" in invalid_result.structured_content['result']

        try:
            # Test non-list input - should raise ToolError
            await client.call_tool("get_shipping_state", {
                "cases": "not a list"
            })
            pytest.fail("Should raise ToolError for non-list input")
        except Exception as e:
            assert "Input validation error: 'not a list' is not of type 'array'" in str(e)
