from typing import Optional
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: Optional[str]
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ReceiptData(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    currency: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    expense_category: Optional[str] = None
    classification_confidence: Optional[float] = None
    document_type: Optional[str] = None
    document_confidence: Optional[float] = None
    reconciliation_notes: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    data: ReceiptData
    changes: list[str] = Field(default_factory=list)
