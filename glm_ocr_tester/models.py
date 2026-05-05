"""Pydantic models for the payslip JSON schema."""

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Employee(BaseModel):
    name: Optional[str] = None
    employee_number: Optional[str] = None


class PayrollPeriod(BaseModel):
    month_year: Optional[str] = None
    correction_number: Optional[str] = None


class EarningLineItem(BaseModel):
    wage_code: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None
    ytd_amount: Optional[float] = None


class DeductionLineItem(BaseModel):
    description: Optional[str] = None
    employee_amount: Optional[float] = None
    employer_amount: Optional[float] = None
    ytd_amount: Optional[float] = None


class LNValue(BaseModel):
    """L (laufend/current) and N (nachfolgend/correction) amounts."""
    L: Optional[float] = None
    N: Optional[float] = None


class TaxDetails(BaseModel):
    lohnsteuer: LNValue = Field(default_factory=LNValue)
    kirchensteuer: LNValue = Field(default_factory=LNValue)
    solidaritaetszuschlag: LNValue = Field(default_factory=LNValue)


class SocialSecurity(BaseModel):
    rentenversicherung: LNValue = Field(default_factory=LNValue)
    arbeitslosenversicherung: LNValue = Field(default_factory=LNValue)
    krankenversicherung: LNValue = Field(default_factory=LNValue)
    pflegeversicherung: LNValue = Field(default_factory=LNValue)


class Earnings(BaseModel):
    line_items: List[EarningLineItem] = Field(default_factory=list)
    total_gross: Optional[float] = None


class Deductions(BaseModel):
    line_items: List[DeductionLineItem] = Field(default_factory=list)
    total_deductions: Optional[float] = None
    total_tax_deductions: LNValue = Field(default_factory=LNValue)
    total_ss_deductions: LNValue = Field(default_factory=LNValue)
    tax_details: TaxDetails = Field(default_factory=TaxDetails)
    social_security: SocialSecurity = Field(default_factory=SocialSecurity)


class NetPay(BaseModel):
    net_amount: Optional[float] = None
    payment_method: Optional[str] = None
    bank_account_iban: Optional[str] = None
    bank_name: Optional[str] = None


class EmployerCosts(BaseModel):
    sv_ag_anteil: Optional[float] = None
    zus_ag_kosten: Optional[float] = None
    gesamtkosten: Optional[float] = None


class YtdAdjustment(BaseModel):
    wage_code: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None


class YtdSummary(BaseModel):
    total_net_ytd: Optional[float] = None
    net_adjustments: List[YtdAdjustment] = Field(default_factory=list)


class ValidationNotes(BaseModel):
    missing_fields: List[str] = Field(default_factory=list)
    ambiguous_fields: List[str] = Field(default_factory=list)
    arithmetic_warnings: List[str] = Field(default_factory=list)


class PayslipSchema(BaseModel):
    document_type: str = "payslip"
    language_detected: Optional[str] = None
    employee: Employee = Field(default_factory=Employee)
    payroll_period: PayrollPeriod = Field(default_factory=PayrollPeriod)
    earnings: Earnings = Field(default_factory=Earnings)
    deductions: Deductions = Field(default_factory=Deductions)
    net_pay: NetPay = Field(default_factory=NetPay)
    employer_costs: EmployerCosts = Field(default_factory=EmployerCosts)
    ytd_summary: YtdSummary = Field(default_factory=YtdSummary)
    validation_notes: ValidationNotes = Field(default_factory=ValidationNotes)

    @classmethod
    def json_schema_text(cls) -> str:
        return (
            "{\n"
            '  "document_type": "payslip",\n'
            '  "language_detected": "",\n'
            '  "employee": {"name": "", "employee_number": ""},\n'
            '  "payroll_period": {"month_year": "", "correction_number": ""},\n'
            '  "earnings": {"line_items": [{"wage_code": "", "description": "", "amount": null, "ytd_amount": null}], "total_gross": null},\n'
            '  "deductions": {"line_items": [{"description": "", "employee_amount": null, "employer_amount": null, "ytd_amount": null}], "total_deductions": null, "total_tax_deductions": {"L": null, "N": null}, "total_ss_deductions": {"L": null, "N": null}, "tax_details": {"lohnsteuer": {"L": null, "N": null}, "kirchensteuer": {"L": null, "N": null}, "solidaritaetszuschlag": {"L": null, "N": null}}, "social_security": {"rentenversicherung": {"L": null, "N": null}, "arbeitslosenversicherung": {"L": null, "N": null}, "krankenversicherung": {"L": null, "N": null}, "pflegeversicherung": {"L": null, "N": null}}},\n'
            '  "net_pay": {"net_amount": null, "payment_method": "", "bank_account_iban": "", "bank_name": ""},\n'
            '  "employer_costs": {"sv_ag_anteil": null, "zus_ag_kosten": null, "gesamtkosten": null},\n'
            '  "ytd_summary": {"total_net_ytd": null, "net_adjustments": []},\n'
            '  "validation_notes": {"missing_fields": [], "ambiguous_fields": [], "arithmetic_warnings": []}\n'
            "}"
        )


class ValidationIssue(BaseModel):
    field: str
    extracted: Any
    expected: Any
    message: str
