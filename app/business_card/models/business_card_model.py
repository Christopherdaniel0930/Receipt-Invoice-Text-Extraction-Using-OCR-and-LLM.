"""Domain model for business card extraction."""

from typing import Optional

from pydantic import BaseModel, ConfigDict


class BusinessCardData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: Optional[str] = None
    designation: Optional[str] = None
    company_name: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
