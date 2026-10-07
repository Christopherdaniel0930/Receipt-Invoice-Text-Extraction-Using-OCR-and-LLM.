"""Combine semantic and deterministic business card fields."""


def merge_business_card(llm_result=None, deterministic_result=None) -> dict:
    llm_result = llm_result or {}
    deterministic_result = deterministic_result or {}
    return {
        "name": llm_result.get("name"),
        "designation": llm_result.get("designation"),
        "company_name": llm_result.get("company_name"),
        "phone": deterministic_result.get("phone"),
        "fax": deterministic_result.get("fax"),
        "email": deterministic_result.get("email"),
        "website": deterministic_result.get("website"),
        "address": deterministic_result.get("address"),
    }
