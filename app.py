import os
import json
import sqlite3
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Integrated Intelligence Platform",
    page_icon="🏢",
    layout="wide"
)

st.title("Integrated Intelligence Platform")
st.caption("Capital partners · Deal pipeline · Investment and leasing comps")


# -----------------------------
# AI config
# -----------------------------
USE_OPENAI_FOR_LOCAL_TESTING = False
USE_MOCK_AI = False


# -----------------------------
# AI helpers
# -----------------------------
def call_claude(prompt: str, api_key_override: str | None = None) -> str:
    api_key = api_key_override or os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return (
            "Claude API key not set. Enter a Claude API key in the sidebar "
            "or set ANTHROPIC_API_KEY as an environment variable."
        )

    if api_key == "TEST_MODE":
        return """
[TEST MODE]

A runtime API key was received successfully.

This confirms:
- the sidebar input is being passed into call_ai()
- call_ai() is passing it into call_claude()
- the app would attempt a Claude call when a real key is provided

No external API call was made.
"""

    if Anthropic is None:
        return "Anthropic package not installed. Run: pip install anthropic"

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1200,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "OpenAI API key not set. Set OPENAI_API_KEY to enable local test outputs."

    if OpenAI is None:
        return "OpenAI package not installed. Run: pip install openai"

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def call_mock_ai(prompt: str) -> str:
    return """
[MOCK AI OUTPUT]

This is a simulated AI response for local demonstration.

Key observations:
- The selected deal has been compared against capital partner mandates.
- Matching considers ticket size, target IRR, sector preference, status and meeting notes.
- Relevant comps are retrieved from the structured comps dataset.
- The recommendation should be reviewed by a human before external use.

This mock response exists so the app remains usable without an API key.
"""


def call_ai(prompt: str, anthropic_api_key: str | None = None) -> str:
    if USE_MOCK_AI:
        return call_mock_ai(prompt)

    if USE_OPENAI_FOR_LOCAL_TESTING:
        return call_openai(prompt)

    return call_claude(prompt, api_key_override=anthropic_api_key)


# -----------------------------
# Seed data
# -----------------------------
capital_partners = [
    {
        "id": "CP001",
        "organisation_name": "NorthBridge Sovereign Capital",
        "key_contact": "James Whitmore",
        "partner_type": "Sovereign Wealth / Institutional Fund",
        "mandate_summary": "Large-ticket core-plus investments in London and major European cities. Focus on stabilised or lightly repositioned office and mixed-use assets.",
        "min_ticket_size_m": 100,
        "max_ticket_size_m": 300,
        "target_irr_min": 10,
        "target_irr_max": 13,
        "preferred_sectors": "Office, Mixed-use",
        "preferred_geographies": "London, Paris, Berlin, Amsterdam",
        "last_meeting_date": "2026-04-12",
        "meeting_notes": "Interested in income-producing assets with credible reversionary upside. They are cautious on heavy refurbishment risk but open to light capex where leasing evidence is strong.",
        "current_pipeline_deals_reviewed": "Holborn Gate, Midtown Chambers",
        "status": "Active",
    },
    {
        "id": "CP002",
        "organisation_name": "Alba European Institutions",
        "key_contact": "Sophie Grant",
        "partner_type": "Sovereign Wealth / Institutional Fund",
        "mandate_summary": "Core-plus pan-European real estate with a focus on defensive income, ESG credentials and prime city locations.",
        "min_ticket_size_m": 100,
        "max_ticket_size_m": 250,
        "target_irr_min": 10,
        "target_irr_max": 12,
        "preferred_sectors": "Office, Mixed-use, Residential-led",
        "preferred_geographies": "London, Madrid, Milan, Stockholm",
        "last_meeting_date": "2026-03-28",
        "meeting_notes": "Strong appetite for London assets where income durability can be demonstrated. They asked specifically for more evidence on occupational demand and recent leasing comps.",
        "current_pipeline_deals_reviewed": "Farringdon Works, Soho Square House",
        "status": "Active",
    },
    {
        "id": "CP003",
        "organisation_name": "Europa Value Partners",
        "key_contact": "Daniel Keller",
        "partner_type": "Pan-European Value-Add Fund",
        "mandate_summary": "Value-add office and mixed-use acquisitions requiring leasing, refurbishment or repositioning. Comfortable with business plans involving moderate capex.",
        "min_ticket_size_m": 20,
        "max_ticket_size_m": 75,
        "target_irr_min": 14,
        "target_irr_max": 18,
        "preferred_sectors": "Office, Mixed-use",
        "preferred_geographies": "London, Berlin, Dublin, Barcelona",
        "last_meeting_date": "2026-04-18",
        "meeting_notes": "Actively looking for Central London office opportunities with vacancy or short WAULT where rents can be grown. They prefer situations with clear leasing comparables.",
        "current_pipeline_deals_reviewed": "Fitzrovia House, Camden Works",
        "status": "Active",
    },
    {
        "id": "CP004",
        "organisation_name": "Crescent European Real Estate",
        "key_contact": "Marta Alvarez",
        "partner_type": "Pan-European Value-Add Fund",
        "mandate_summary": "Targets £20m-£75m value-add investments in office and mixed-use assets, especially where active asset management can drive returns.",
        "min_ticket_size_m": 20,
        "max_ticket_size_m": 75,
        "target_irr_min": 14,
        "target_irr_max": 18,
        "preferred_sectors": "Office, Mixed-use, Flex",
        "preferred_geographies": "London, Manchester, Paris, Lisbon",
        "last_meeting_date": "2026-04-08",
        "meeting_notes": "They are keen on assets with repositioning potential and a credible exit story. They are currently avoiding very large lot sizes and speculative development exposure.",
        "current_pipeline_deals_reviewed": "Shoreditch Exchange, Holborn Gate",
        "status": "Active",
    },
    {
        "id": "CP005",
        "organisation_name": "Harrington Family Office",
        "key_contact": "Edward Harrington",
        "partner_type": "Family Office",
        "mandate_summary": "Flexible, relationship-led capital for opportunistic real estate investments. Prefers smaller club deals with strong downside protection.",
        "min_ticket_size_m": 10,
        "max_ticket_size_m": 40,
        "target_irr_min": 13,
        "target_irr_max": 20,
        "preferred_sectors": "Office, Residential-led, Mixed-use",
        "preferred_geographies": "London and South East England",
        "last_meeting_date": "2026-04-21",
        "meeting_notes": "Open to opportunistic situations where pricing is attractive. They like off-market or relationship-driven deals and are flexible on structure.",
        "current_pipeline_deals_reviewed": "Camden Works, Wandsworth Yard",
        "status": "Active",
    },
    {
        "id": "CP006",
        "organisation_name": "Lydian Private Capital",
        "key_contact": "Amelia Chen",
        "partner_type": "Family Office",
        "mandate_summary": "Family office focused on flexible equity investments between £10m and £40m. Interested in special situations and development-backed value creation.",
        "min_ticket_size_m": 10,
        "max_ticket_size_m": 40,
        "target_irr_min": 14,
        "target_irr_max": 22,
        "preferred_sectors": "Residential-led, Mixed-use, Flex",
        "preferred_geographies": "London, Oxford, Cambridge",
        "last_meeting_date": "2026-03-19",
        "meeting_notes": "They are open-minded on sector but need conviction around planning, exit liquidity and sponsor alignment. They prefer co-investment structures.",
        "current_pipeline_deals_reviewed": "Deptford Living, Wandsworth Yard",
        "status": "Passive",
    },
    {
        "id": "CP007",
        "organisation_name": "Hudson Gate Capital",
        "key_contact": "Michael Reynolds",
        "partner_type": "US Private Equity Fund",
        "mandate_summary": "US private equity capital targeting London gateway assets with control positions and high-return business plans.",
        "min_ticket_size_m": 40,
        "max_ticket_size_m": 150,
        "target_irr_min": 18,
        "target_irr_max": 25,
        "preferred_sectors": "Office, Mixed-use",
        "preferred_geographies": "Central London",
        "last_meeting_date": "2026-04-04",
        "meeting_notes": "They are looking for control-oriented opportunities with clear distress, vacancy or repositioning angles. They are not interested in passive minority stakes.",
        "current_pipeline_deals_reviewed": "Fitzrovia House, Soho Square House",
        "status": "Active",
    },
    {
        "id": "CP008",
        "organisation_name": "Atlantic Ridge Partners",
        "key_contact": "Rachel Stein",
        "partner_type": "US Private Equity Fund",
        "mandate_summary": "Majority-control investor focused on London gateway assets and opportunistic office repositioning.",
        "min_ticket_size_m": 50,
        "max_ticket_size_m": 180,
        "target_irr_min": 18,
        "target_irr_max": 24,
        "preferred_sectors": "Office, Flex, Mixed-use",
        "preferred_geographies": "London, New York, Boston",
        "last_meeting_date": "2026-04-15",
        "meeting_notes": "Interested in larger London assets where pricing has reset. They asked for opportunities with strong evidence of tenant demand and an executable capex plan.",
        "current_pipeline_deals_reviewed": "Shoreditch Exchange, Midtown Chambers",
        "status": "Active",
    },
    {
        "id": "CP009",
        "organisation_name": "Meridian Credit Partners",
        "key_contact": "Oliver Hayes",
        "partner_type": "Debt / Preferred Equity Provider",
        "mandate_summary": "Provides senior and mezzanine finance for UK commercial real estate up to 65% LTV.",
        "min_ticket_size_m": 15,
        "max_ticket_size_m": 50,
        "target_irr_min": 8,
        "target_irr_max": 14,
        "preferred_sectors": "Office, Mixed-use, Residential-led",
        "preferred_geographies": "UK major cities",
        "last_meeting_date": "2026-04-10",
        "meeting_notes": "They are actively lending but cautious on speculative vacancy. They want strong sponsors, conservative leverage and clear evidence of exit or refinance routes.",
        "current_pipeline_deals_reviewed": "Deptford Living, Holborn Gate",
        "status": "Active",
    },
    {
        "id": "CP010",
        "organisation_name": "Cavendish Preferred Capital",
        "key_contact": "Priya Nair",
        "partner_type": "Debt / Preferred Equity Provider",
        "mandate_summary": "Preferred equity and mezzanine provider for transitional real estate situations. Ticket sizes from £15m to £50m.",
        "min_ticket_size_m": 15,
        "max_ticket_size_m": 50,
        "target_irr_min": 10,
        "target_irr_max": 16,
        "preferred_sectors": "Office, Residential-led, Mixed-use, Flex",
        "preferred_geographies": "London and UK regional cities",
        "last_meeting_date": "2026-03-30",
        "meeting_notes": "They are open to transitional assets where the business plan is already well progressed. They prefer to sit behind conservative senior debt rather than take full equity risk.",
        "current_pipeline_deals_reviewed": "Wandsworth Yard, Farringdon Works",
        "status": "On hold",
    },
]

deal_pipeline = [
    {
        "id": "D001",
        "address": "Fitzrovia House, 42-48 Charlotte Street, London",
        "postcode": "W1T 2NS",
        "sector": "Office",
        "nia_sqft": 68500,
        "asking_price_m": 72.5,
        "initial_yield": 4.35,
        "underwrite_irr": 17.2,
        "equity_requirement_m": 34,
        "stage": "In diligence",
        "key_dates": "NBO submitted 2026-04-05; IC review 2026-05-08; exclusivity expiry 2026-05-22",
        "deal_summary": "Central London office asset with short WAULT and refurbishment-led reversion. Current passing rent is below recent leasing evidence in Fitzrovia. Business plan assumes targeted capex, improved amenities and phased leasing to creative and media tenants.",
    },
    {
        "id": "D002",
        "address": "Holborn Gate, 88 High Holborn, London",
        "postcode": "WC1V 6LJ",
        "sector": "Office",
        "nia_sqft": 112000,
        "asking_price_m": 128,
        "initial_yield": 5.1,
        "underwrite_irr": 12.4,
        "equity_requirement_m": 58,
        "stage": "Under offer",
        "key_dates": "Offer accepted 2026-04-14; DD pack received 2026-04-19; target exchange 2026-06-10",
        "deal_summary": "Large income-producing office asset in Holborn with stable existing income and modest reversion. The opportunity suits core-plus capital seeking London exposure. Key diligence focus is lease expiry concentration and capex required for upcoming refurbishments.",
    },
    {
        "id": "D003",
        "address": "Midtown Chambers, 16 Kingsway, London",
        "postcode": "WC2B 6UN",
        "sector": "Office",
        "nia_sqft": 54000,
        "asking_price_m": 61,
        "initial_yield": 4.75,
        "underwrite_irr": 15.8,
        "equity_requirement_m": 29,
        "stage": "Sourced",
        "key_dates": "Teaser received 2026-04-22; initial review 2026-04-25; site visit pending",
        "deal_summary": "Midtown office opportunity with vacancy on upper floors and potential to capture rental growth through refurbishment. Pricing appears full but leasing evidence nearby is improving. Suitable for value-add capital with appetite for moderate execution risk.",
    },
    {
        "id": "D004",
        "address": "Soho Square House, 9 Soho Square, London",
        "postcode": "W1D 3QF",
        "sector": "Mixed-use",
        "nia_sqft": 43500,
        "asking_price_m": 58,
        "initial_yield": 3.9,
        "underwrite_irr": 18.6,
        "equity_requirement_m": 31,
        "stage": "Under offer",
        "key_dates": "Second-round bid 2026-04-11; vendor interview 2026-04-20; preferred bidder decision pending",
        "deal_summary": "Mixed-use office and retail asset in Soho with below-market retail rents and short office leases. The business plan depends on active leasing, rent reviews and selective refurbishment. High return potential but execution and entry yield require careful scrutiny.",
    },
    {
        "id": "D005",
        "address": "Camden Works, 101 Camden Road, London",
        "postcode": "NW1 9HA",
        "sector": "Mixed-use",
        "nia_sqft": 39000,
        "asking_price_m": 32,
        "initial_yield": 5.6,
        "underwrite_irr": 16.9,
        "equity_requirement_m": 15,
        "stage": "Passed",
        "key_dates": "Reviewed 2026-03-12; passed 2026-03-19",
        "deal_summary": "Office and retail mixed-use asset with attractive headline yield but weak tenant covenant profile. The team passed due to limited rental evidence and higher-than-expected capex. Could be revisited if pricing softens materially.",
    },
    {
        "id": "D006",
        "address": "Deptford Living, Creekside, London",
        "postcode": "SE8 3DX",
        "sector": "Residential-led Development",
        "nia_sqft": 146000,
        "asking_price_m": 42,
        "initial_yield": 0.0,
        "underwrite_irr": 19.5,
        "equity_requirement_m": 24,
        "stage": "In diligence",
        "key_dates": "Heads of terms agreed 2026-04-03; planning review 2026-04-17; IC date 2026-05-13",
        "deal_summary": "Residential-led development site with consented scheme and ground-floor commercial space. Returns are driven by planning optimisation, build cost control and exit pricing. Key risks include construction inflation, affordable housing obligations and sales absorption.",
    },
    {
        "id": "D007",
        "address": "Wandsworth Yard, Armoury Way, London",
        "postcode": "SW18 1TH",
        "sector": "Residential-led Development",
        "nia_sqft": 98000,
        "asking_price_m": 35,
        "initial_yield": 0.0,
        "underwrite_irr": 18.1,
        "equity_requirement_m": 20,
        "stage": "Exchanged",
        "key_dates": "Exchanged 2026-04-09; completion due 2026-05-31; planning amendments due 2026-06-28",
        "deal_summary": "Residential-led site with an exchanged position and potential to improve unit mix. Business plan assumes planning amendments and a phased sales strategy. The deal is suitable for opportunistic capital comfortable with development risk.",
    },
    {
        "id": "D008",
        "address": "Shoreditch Exchange, Great Eastern Street, London",
        "postcode": "EC2A 3HU",
        "sector": "Flex / Co-working",
        "nia_sqft": 47000,
        "asking_price_m": 44,
        "initial_yield": 4.2,
        "underwrite_irr": 18.8,
        "equity_requirement_m": 22,
        "stage": "Sourced",
        "key_dates": "Opportunity received 2026-04-23; operator meeting 2026-04-30; initial IC pending",
        "deal_summary": "Flex workspace opportunity in Shoreditch with partial occupancy and potential operator-led repositioning. The return profile depends on stabilising desk occupancy and improving amenity provision. Suitable for higher-return capital but sensitive to occupational demand assumptions.",
    },
    {
        "id": "D009",
        "address": "Paddington Flex Hub, 19 Eastbourne Terrace, London",
        "postcode": "W2 6LG",
        "sector": "Flex / Co-working",
        "nia_sqft": 52000,
        "asking_price_m": 49,
        "initial_yield": 4.0,
        "underwrite_irr": 13.7,
        "equity_requirement_m": 21,
        "stage": "Passed",
        "key_dates": "Reviewed 2026-02-18; operator diligence 2026-02-25; passed 2026-03-02",
        "deal_summary": "Flex office asset with good transport links but weak operator covenant and limited pricing discount. The team passed because the risk-adjusted return was not compelling. Further review may be warranted if the vendor revises price expectations.",
    },
    {
        "id": "D010",
        "address": "Farringdon Works, 25-31 Clerkenwell Road, London",
        "postcode": "EC1M 5PA",
        "sector": "Office",
        "nia_sqft": 76000,
        "asking_price_m": 86,
        "initial_yield": 5.25,
        "underwrite_irr": 11.8,
        "equity_requirement_m": 39,
        "stage": "Completed",
        "key_dates": "Completed 2026-01-30; asset management plan approved 2026-03-15",
        "deal_summary": "Completed acquisition of a stabilised Farringdon office asset with diversified tenant base and near-term rent review upside. The investment thesis is defensive income with moderate rental growth. Current focus is executing the asset management plan and monitoring lease events.",
    },
]

investment_and_leasing_comps = [
    {
        "id": "C001",
        "comp_type": "Investment",
        "address": "31-35 Foley Street, London",
        "postcode": "W1W 7TS",
        "transaction_date": "2025-11-14",
        "sector": "Office",
        "size_sqft": 62000,
        "price_psf": 1040,
        "niy": 4.45,
        "purchaser_type": "Private Equity Fund",
        "headline_rent_psf": None,
        "lease_length_years": None,
        "rent_free_months": None,
        "tenant_sector": None,
        "source": "Broker investment note",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C002",
        "comp_type": "Investment",
        "address": "33 Foley Street, London",
        "postcode": "W1W 7TS",
        "transaction_date": "2025-11-16",
        "sector": "Office",
        "size_sqft": 61800,
        "price_psf": 1038,
        "niy": 4.46,
        "purchaser_type": "Private Equity Fund",
        "headline_rent_psf": None,
        "lease_length_years": None,
        "rent_free_months": None,
        "tenant_sector": None,
        "source": "Market press report",
        "confidence_level": "Medium",
        "duplicate_flag": True,
    },
    {
        "id": "C003",
        "comp_type": "Investment",
        "address": "12 Red Lion Square, London",
        "postcode": "WC1R 4HQ",
        "transaction_date": "2025-09-22",
        "sector": "Office",
        "size_sqft": 88000,
        "price_psf": 960,
        "niy": 5.05,
        "purchaser_type": "Institutional Fund",
        "headline_rent_psf": None,
        "lease_length_years": None,
        "rent_free_months": None,
        "tenant_sector": None,
        "source": "Agent sales evidence",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C004",
        "comp_type": "Investment",
        "address": "70 Broadwick Street, London",
        "postcode": "W1F 9QZ",
        "transaction_date": "2024-12-05",
        "sector": "Mixed-use",
        "size_sqft": 41000,
        "price_psf": 1185,
        "niy": 3.85,
        "purchaser_type": "Family Office",
        "headline_rent_psf": None,
        "lease_length_years": None,
        "rent_free_months": None,
        "tenant_sector": None,
        "source": "Capital markets update",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C005",
        "comp_type": "Investment",
        "address": "18 Great Eastern Street, London",
        "postcode": "EC2A 3EJ",
        "transaction_date": "2025-06-18",
        "sector": "Flex / Co-working",
        "size_sqft": 50500,
        "price_psf": 890,
        "niy": 4.65,
        "purchaser_type": "US Private Equity Fund",
        "headline_rent_psf": None,
        "lease_length_years": None,
        "rent_free_months": None,
        "tenant_sector": None,
        "source": "Internal market tracker",
        "confidence_level": "Medium",
        "duplicate_flag": False,
    },
    {
        "id": "C006",
        "comp_type": "Leasing",
        "address": "20 Rathbone Place, London",
        "postcode": "W1T 1HY",
        "transaction_date": "2025-10-03",
        "sector": "Office",
        "size_sqft": 18500,
        "price_psf": None,
        "niy": None,
        "purchaser_type": None,
        "headline_rent_psf": 92.5,
        "lease_length_years": 10,
        "rent_free_months": 18,
        "tenant_sector": "Media",
        "source": "Leasing agent update",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C007",
        "comp_type": "Leasing",
        "address": "22 Rathbone Place, London",
        "postcode": "W1T 1HY",
        "transaction_date": "2025-10-05",
        "sector": "Office",
        "size_sqft": 18400,
        "price_psf": None,
        "niy": None,
        "purchaser_type": None,
        "headline_rent_psf": 92.0,
        "lease_length_years": 10,
        "rent_free_months": 18,
        "tenant_sector": "Media",
        "source": "Occupier market report",
        "confidence_level": "Medium",
        "duplicate_flag": True,
    },
    {
        "id": "C008",
        "comp_type": "Leasing",
        "address": "5 Chancery Lane, London",
        "postcode": "WC2A 1LG",
        "transaction_date": "2025-07-11",
        "sector": "Office",
        "size_sqft": 24000,
        "price_psf": None,
        "niy": None,
        "purchaser_type": None,
        "headline_rent_psf": 78.0,
        "lease_length_years": 7,
        "rent_free_months": 12,
        "tenant_sector": "Legal",
        "source": "Lease advisory note",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C009",
        "comp_type": "Leasing",
        "address": "15 Great Eastern Street, London",
        "postcode": "EC2A 3HU",
        "transaction_date": "2025-12-09",
        "sector": "Office",
        "size_sqft": 16000,
        "price_psf": None,
        "niy": None,
        "purchaser_type": None,
        "headline_rent_psf": 72.5,
        "lease_length_years": 5,
        "rent_free_months": 10,
        "tenant_sector": "Technology",
        "source": "Broker leasing bulletin",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
    {
        "id": "C010",
        "comp_type": "Leasing",
        "address": "40 Clerkenwell Road, London",
        "postcode": "EC1M 5PS",
        "transaction_date": "2025-08-21",
        "sector": "Office",
        "size_sqft": 21000,
        "price_psf": None,
        "niy": None,
        "purchaser_type": None,
        "headline_rent_psf": 76.0,
        "lease_length_years": 8,
        "rent_free_months": 14,
        "tenant_sector": "Professional Services",
        "source": "Internal leasing tracker",
        "confidence_level": "High",
        "duplicate_flag": False,
    },
]

# -----------------------------
# SQLite persistence
# -----------------------------
DB_PATH = "integrated_intelligence.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialise_database():
    conn = get_connection()

    existing_tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';",
        conn
    )["name"].tolist()

    if "capital_partners" not in existing_tables:
        pd.DataFrame(capital_partners).to_sql(
            "capital_partners",
            conn,
            index=False
        )

    if "deal_pipeline" not in existing_tables:
        pd.DataFrame(deal_pipeline).to_sql(
            "deal_pipeline",
            conn,
            index=False
        )

    if "comps" not in existing_tables:
        pd.DataFrame(investment_and_leasing_comps).to_sql(
            "comps",
            conn,
            index=False
        )

    conn.close()


def load_data():
    conn = get_connection()

    partners = pd.read_sql("SELECT * FROM capital_partners", conn)
    deals = pd.read_sql("SELECT * FROM deal_pipeline", conn)
    comps = pd.read_sql("SELECT * FROM comps", conn)

    conn.close()

    return partners, deals, comps


initialise_database()
partners_df, deals_df, comps_df = load_data()


# -----------------------------
# Prototype access control
# -----------------------------
ROLE_PERMISSIONS = {
    "Investment Team": {
        "view_deals": True,
        "view_partners": True,
        "view_comps": True,
        "view_sensitive_partner_notes": True,
        "generate_ai": True,
        "download_reports": True,
    },
    "Capital Markets": {
        "view_deals": True,
        "view_partners": True,
        "view_comps": True,
        "view_sensitive_partner_notes": True,
        "generate_ai": True,
        "download_reports": True,
    },
    "Asset Manager": {
        "view_deals": True,
        "view_partners": False,
        "view_comps": True,
        "view_sensitive_partner_notes": False,
        "generate_ai": True,
        "download_reports": True,
    },
    "Fund Administrator": {
        "view_deals": True,
        "view_partners": False,
        "view_comps": False,
        "view_sensitive_partner_notes": False,
        "generate_ai": False,
        "download_reports": False,
    }
}


def can(role: str, permission: str) -> bool:
    return ROLE_PERMISSIONS.get(role, {}).get(permission, False)


def access_denied_message(resource_name: str) -> None:
    st.warning(
        f"Access restricted: your selected prototype role does not have permission to view {resource_name}."
    )


def redact_partner_data(df: pd.DataFrame, role: str) -> pd.DataFrame:
    if can(role, "view_sensitive_partner_notes"):
        return df

    redacted = df.copy()
    sensitive_columns = ["meeting_notes", "current_pipeline_deals_reviewed"]

    for col in sensitive_columns:
        if col in redacted.columns:
            redacted[col] = "[Restricted by role]"

    return redacted


def redact_ai_context_for_role(
    role: str,
    selected_deal: dict,
    selected_partner: dict | None,
    relevant_comps: pd.DataFrame,
    matches: pd.DataFrame
) -> tuple[dict, dict | None, pd.DataFrame, pd.DataFrame]:
    safe_deal = selected_deal.copy()
    safe_partner = selected_partner.copy() if selected_partner else None
    safe_comps = relevant_comps.copy()
    safe_matches = matches.copy()

    if not can(role, "view_sensitive_partner_notes"):
        if safe_partner:
            safe_partner["meeting_notes"] = "[Restricted by role]"
            safe_partner["current_pipeline_deals_reviewed"] = "[Restricted by role]"

        if "meeting_notes" in safe_matches.columns:
            safe_matches["meeting_notes"] = "[Restricted by role]"

    if not can(role, "view_partners"):
        safe_partner = None
        partner_cols_to_redact = [
            "organisation_name",
            "key_contact",
            "meeting_notes",
            "current_pipeline_deals_reviewed"
        ]

        for col in partner_cols_to_redact:
            if col in safe_matches.columns:
                safe_matches[col] = "[Restricted by role]"

    if not can(role, "view_comps"):
        safe_comps = pd.DataFrame()

    return safe_deal, safe_partner, safe_comps, safe_matches


# -----------------------------
# Duplicate detection
# -----------------------------
def normalise_text(value: str) -> str:
    if pd.isna(value):
        return ""

    value = str(value).lower()
    replacements = {
        ",": "",
        ".": "",
        "-": " ",
        "&": "and",
        "street": "st",
        "road": "rd",
        "place": "pl",
        "london": ""
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    return " ".join(value.split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise_text(a), normalise_text(b)).ratio()


def numeric_similarity(a, b, tolerance: float) -> float:
    if pd.isna(a) or pd.isna(b) or a in [None, 0] or b in [None, 0]:
        return 0.0

    diff_ratio = abs(float(a) - float(b)) / max(float(a), float(b))

    if diff_ratio <= tolerance:
        return 1.0

    if diff_ratio <= tolerance * 2:
        return 0.5

    return 0.0


def date_proximity_score(date_a, date_b, tolerance_days: int = 30) -> float:
    try:
        d1 = pd.to_datetime(date_a)
        d2 = pd.to_datetime(date_b)
    except Exception:
        return 0.0

    days = abs((d1 - d2).days)

    if days <= tolerance_days:
        return 1.0

    if days <= tolerance_days * 2:
        return 0.5

    return 0.0


def calculate_duplicate_score(row_a: dict, row_b: dict) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    address_score = similarity(row_a.get("address", ""), row_b.get("address", ""))
    if address_score >= 0.72:
        score += 30
        reasons.append("Similar address")

    if row_a.get("postcode") == row_b.get("postcode"):
        score += 25
        reasons.append("Same postcode")

    if row_a.get("sector") == row_b.get("sector"):
        score += 10
        reasons.append("Same sector")

    if row_a.get("comp_type") == row_b.get("comp_type"):
        score += 10
        reasons.append("Same comp type")

    date_score = date_proximity_score(
        row_a.get("transaction_date"),
        row_b.get("transaction_date")
    )
    if date_score == 1:
        score += 15
        reasons.append("Transaction dates within 30 days")
    elif date_score == 0.5:
        score += 7
        reasons.append("Transaction dates within 60 days")

    size_score = numeric_similarity(
        row_a.get("size_sqft"),
        row_b.get("size_sqft"),
        tolerance=0.05
    )
    if size_score == 1:
        score += 10
        reasons.append("Size within 5%")
    elif size_score == 0.5:
        score += 5
        reasons.append("Size within 10%")

    for metric in ["price_psf", "niy", "headline_rent_psf"]:
        metric_score = numeric_similarity(
            row_a.get(metric),
            row_b.get(metric),
            tolerance=0.03
        )

        if metric_score == 1:
            score += 10
            reasons.append(f"{metric} within 3%")
        elif metric_score == 0.5:
            score += 5
            reasons.append(f"{metric} within 6%")

    return min(score, 100), reasons


def detect_duplicate_comps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["auto_duplicate_flag"] = False
    df["duplicate_group"] = ""
    df["duplicate_confidence"] = 0
    df["duplicate_reason"] = ""

    duplicate_pairs = []
    group_number = 1

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            row_a = df.iloc[i].to_dict()
            row_b = df.iloc[j].to_dict()

            score, reasons = calculate_duplicate_score(row_a, row_b)

            if score >= 70:
                group_id = f"DUP-{group_number:02d}"

                df.at[df.index[i], "auto_duplicate_flag"] = True
                df.at[df.index[j], "auto_duplicate_flag"] = True

                existing_group_a = df.at[df.index[i], "duplicate_group"]
                existing_group_b = df.at[df.index[j], "duplicate_group"]

                if not existing_group_a and not existing_group_b:
                    df.at[df.index[i], "duplicate_group"] = group_id
                    df.at[df.index[j], "duplicate_group"] = group_id
                    group_number += 1
                elif existing_group_a:
                    df.at[df.index[j], "duplicate_group"] = existing_group_a
                elif existing_group_b:
                    df.at[df.index[i], "duplicate_group"] = existing_group_b

                df.at[df.index[i], "duplicate_confidence"] = max(
                    df.at[df.index[i], "duplicate_confidence"],
                    score
                )
                df.at[df.index[j], "duplicate_confidence"] = max(
                    df.at[df.index[j], "duplicate_confidence"],
                    score
                )

                reason_text = "; ".join(reasons)
                df.at[df.index[i], "duplicate_reason"] = reason_text
                df.at[df.index[j], "duplicate_reason"] = reason_text

                duplicate_pairs.append({
                    "comp_a_id": row_a["id"],
                    "comp_a_address": row_a["address"],
                    "comp_b_id": row_b["id"],
                    "comp_b_address": row_b["address"],
                    "duplicate_score": score,
                    "reasons": reason_text
                })

    return df, pd.DataFrame(duplicate_pairs)


comps_df, duplicate_pairs_df = detect_duplicate_comps(comps_df)


# -----------------------------
# Partner matching
# -----------------------------
def calculate_partner_match(deal: dict, partner: dict) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    equity = deal["equity_requirement_m"]
    irr = deal["underwrite_irr"]
    sector = deal["sector"].lower()
    stage = deal["stage"]
    preferred_sectors = partner["preferred_sectors"].lower()

    if partner["status"] == "Active":
        score += 20
        reasons.append("Partner is currently active.")
    elif partner["status"] == "Passive":
        score += 8
        reasons.append("Partner is passive but not unavailable.")
    else:
        score -= 10
        reasons.append("Partner is currently on hold.")

    if partner["min_ticket_size_m"] <= equity <= partner["max_ticket_size_m"]:
        score += 30
        reasons.append("Equity requirement sits within mandate ticket size.")
    elif abs(equity - partner["min_ticket_size_m"]) <= 10 or abs(equity - partner["max_ticket_size_m"]) <= 10:
        score += 12
        reasons.append("Equity requirement is close to ticket size range.")
    else:
        score -= 10
        reasons.append("Equity requirement is outside preferred ticket size.")

    if partner["target_irr_min"] <= irr <= partner["target_irr_max"]:
        score += 25
        reasons.append("Underwrite IRR fits target return range.")
    elif irr >= partner["target_irr_min"] - 2:
        score += 10
        reasons.append("Underwrite IRR is close to return target.")
    else:
        score -= 10
        reasons.append("Underwrite IRR may be below return target.")

    if sector in preferred_sectors:
        score += 20
        reasons.append("Sector aligns with preferred sectors.")
    elif "office" in sector and "mixed-use" in preferred_sectors:
        score += 8
        reasons.append("Sector has partial overlap with mandate.")
    else:
        score -= 5
        reasons.append("Sector is not an obvious mandate fit.")

    if stage in ["Passed", "Completed"]:
        score -= 25
        reasons.append("Deal is not currently an active fundraising opportunity.")

    return max(0, min(100, score)), reasons


def get_partner_matches(deal_id: str) -> pd.DataFrame:
    deal = deals_df[deals_df["id"] == deal_id].iloc[0].to_dict()
    rows = []

    for _, partner_row in partners_df.iterrows():
        partner = partner_row.to_dict()
        score, reasons = calculate_partner_match(deal, partner)

        rows.append({
            "partner_id": partner["id"],
            "organisation_name": partner["organisation_name"],
            "key_contact": partner["key_contact"],
            "partner_type": partner["partner_type"],
            "status": partner["status"],
            "match_score": score,
            "reasons": " ".join(reasons),
            "meeting_notes": partner["meeting_notes"]
        })

    return pd.DataFrame(rows).sort_values("match_score", ascending=False)


def get_relevant_comps(deal: dict) -> pd.DataFrame:
    sector = deal["sector"]

    if sector == "Office":
        relevant = comps_df[comps_df["sector"].isin(["Office"])]
    elif sector == "Mixed-use":
        relevant = comps_df[comps_df["sector"].isin(["Mixed-use", "Office"])]
    elif "Flex" in sector:
        relevant = comps_df[comps_df["sector"].isin(["Flex / Co-working", "Office"])]
    elif "Residential" in sector:
        relevant = comps_df[comps_df["sector"].isin(["Mixed-use", "Office"])]
    else:
        relevant = comps_df.copy()

    return relevant.sort_values("transaction_date", ascending=False)


# -----------------------------
# AI outputs
# -----------------------------
def generate_meeting_brief(
    partner_id: str,
    user_role: str,
    anthropic_api_key: str | None = None
) -> str:
    if not can(user_role, "generate_ai"):
        return "Access restricted: your role does not have permission to generate AI outputs."

    if not can(user_role, "view_partners"):
        return "Access restricted: your role does not have permission to generate partner meeting briefs."

    partner = partners_df[partners_df["id"] == partner_id].iloc[0].to_dict()

    candidate_deals = []
    for _, deal_row in deals_df.iterrows():
        deal = deal_row.to_dict()

        if deal["stage"] not in ["Passed", "Completed"]:
            score, reasons = calculate_partner_match(deal, partner)

            if score >= 45:
                candidate_deals.append({
                    "deal": deal,
                    "match_score": score,
                    "reasons": reasons
                })

    candidate_deals = sorted(
        candidate_deals,
        key=lambda x: x["match_score"],
        reverse=True
    )[:3]

    prompt = f"""
You are an AI assistant for a real estate investment and asset management firm.

Task:
Generate a concise meeting brief for the selected capital partner.

Rules:
- Use only the provided data.
- Do not invent facts, deals, names, dates, market evidence or metrics.
- If information is missing, write: "Not available from provided data."
- Separate factual observations from judgement calls.
- Every deal recommendation must include the evidence used: ticket size, IRR, sector, status and meeting-note relevance.
- Be concise and suitable for an internal investment meeting.

Capital partner:
{json.dumps(partner, indent=2)}

Relevant pipeline deals:
{json.dumps(candidate_deals, indent=2)}

Output:
1. Partner profile
2. Current appetite based on mandate and meeting notes
3. Deals to raise
   - Deal name/address
   - Match score
   - Why it fits
   - Evidence used
   - Caveats
4. Recent meeting-note implications
5. Suggested questions for the meeting
6. Follow-up actions
"""

    return call_ai(prompt, anthropic_api_key)


def generate_comps_report(
    deal_id: str,
    user_role: str,
    anthropic_api_key: str | None = None
) -> str:
    if not can(user_role, "generate_ai"):
        return "Access restricted: your role does not have permission to generate AI outputs."

    if not can(user_role, "view_deals") or not can(user_role, "view_comps"):
        return "Access restricted: your role does not have permission to generate comps reports."

    deal = deals_df[deals_df["id"] == deal_id].iloc[0].to_dict()
    relevant_comps = get_relevant_comps(deal).head(6).to_dict(orient="records")

    prompt = f"""
You are preparing an external-facing real estate comps report for a real estate investment team.

Task:
Generate a concise, professional comps report for the selected pipeline deal.

Rules:
- Use only the deal and comps data provided.
- Do not invent market data, transactions, rents, yields, dates or purchaser names.
- If information is missing, write: "Not available from provided data."
- Flag any comp where auto_duplicate_flag or duplicate_flag is true and do not double-count it in conclusions.
- Separate investment comps from leasing comps.
- Make clear where the evidence is strong, weak or limited.
- Keep the tone suitable for external circulation.

Pipeline deal:
{json.dumps(deal, indent=2)}

Relevant comps:
{json.dumps(relevant_comps, indent=2)}

Output:
1. Executive summary
2. Deal overview
3. Investment comps
   - Address
   - Date
   - Size
   - Price psf
   - NIY
   - Purchaser type
   - Relevance to subject deal
4. Leasing comps
   - Address
   - Date
   - Size
   - Headline rent psf
   - Lease length
   - Rent-free period
   - Tenant sector
   - Relevance to subject deal
5. Benchmarking commentary
   - Pricing/yield evidence
   - Leasing/rental evidence
   - Any duplicate records excluded or caveated
6. Key risks and caveats
7. Source records used by ID
"""

    return call_ai(prompt, anthropic_api_key)


def generate_investment_analysis(
    deal_id: str,
    user_role: str,
    anthropic_api_key: str | None = None
) -> str:
    if not can(user_role, "generate_ai"):
        return "Access restricted: your role does not have permission to generate AI outputs."

    if not can(user_role, "view_deals"):
        return "Access restricted: your role does not have permission to generate investment analysis."

    deal = deals_df[deals_df["id"] == deal_id].iloc[0].to_dict()

    if can(user_role, "view_comps"):
        relevant_comps = get_relevant_comps(deal).head(6).to_dict(orient="records")
    else:
        relevant_comps = []

    if can(user_role, "view_partners"):
        matches = get_partner_matches(deal_id).head(5).to_dict(orient="records")
    else:
        matches = []

    prompt = f"""
You are preparing an initial investment analysis for a real estate investment team.

Task:
Create a first-pass investment analysis for the selected pipeline deal.

Rules:
- Use only the provided data.
- Do not invent facts, assumptions, market data, financial metrics or investor appetite.
- If information is missing, write: "Not available from provided data."
- Separate facts, interpretation, risks and open questions.
- Do not present capital partner matches as confirmed investor interest.
- Do not make a final investment recommendation; provide a preliminary view only.
- Be concise, analytical and commercially realistic.

Deal:
{json.dumps(deal, indent=2)}

Relevant comps:
{json.dumps(relevant_comps, indent=2)}

Potential capital partners:
{json.dumps(matches, indent=2)}

Output:
1. Deal summary
2. Factual underwriting snapshot
   - Price
   - Initial yield
   - Underwrite IRR
   - Equity requirement
   - Stage
3. Market evidence from comps
   - Supportive evidence
   - Weak or missing evidence
   - Duplicate comp caveats
4. Capital partner fit
   - Strongest potential partners
   - Why they fit
   - Caveats
5. Key risks
6. Questions for further diligence
7. Preliminary view
   - Positive indicators
   - Concerns
   - What would need to be confirmed before progressing
"""

    return call_ai(prompt, anthropic_api_key)


# -----------------------------
# Natural language routing
# -----------------------------
def extract_number(query: str, default: float) -> float:
    for token in query.replace("%", "").replace(">", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return default


def classify_query_intent(query: str) -> str:
    q = query.lower()

    if "duplicate" in q and "comp" in q:
        return "duplicate_comps"

    if "fitzrovia" in q or "w1t" in q:
        return "fitzrovia"

    if ("call" in q or "speak" in q or "contact" in q) and ("week" in q or "partner" in q):
        return "call_this_week"

    if "comps" in q and ("this deal" in q or "selected deal" in q or "report" in q):
        return "comps_for_selected_deal"

    if "partner" in q and "irr" in q:
        return "partner_irr"

    if "deal" in q and "irr" in q and ("above" in q or "over" in q or ">" in q):
        return "high_irr_deals"

    if "office" in q and "deal" in q:
        return "office_deals"

    if "passed" in q and "deal" in q:
        return "passed_deals"

    return "unknown"


def run_structured_query(query: str, selected_deal_id: str, user_role: str) -> tuple[str, pd.DataFrame | str]:
    q = query.lower()
    intent = classify_query_intent(query)

    if intent == "partner_irr":
        if not can(user_role, "view_partners"):
            return "Access restricted.", "Your role does not have permission to query capital partner records."

        threshold = extract_number(q, 15)
        result = partners_df[partners_df["target_irr_max"] >= threshold][[
            "organisation_name",
            "partner_type",
            "target_irr_min",
            "target_irr_max",
            "preferred_sectors",
            "status"
        ]]
        return f"Capital partners with target IRR capacity at or above {threshold}%.", result

    if intent == "office_deals":
        if not can(user_role, "view_deals"):
            return "Access restricted.", "Your role does not have permission to query deal records."

        result = deals_df[deals_df["sector"] == "Office"][[
            "address",
            "sector",
            "asking_price_m",
            "underwrite_irr",
            "equity_requirement_m",
            "stage"
        ]]
        return "Office deals in the pipeline.", result

    if intent == "duplicate_comps":
        if not can(user_role, "view_comps"):
            return "Access restricted.", "Your role does not have permission to query comps records."

        result = comps_df[
            (comps_df["auto_duplicate_flag"] == True) |
            (comps_df.get("duplicate_flag", False) == True)
        ][[
            "id",
            "address",
            "postcode",
            "transaction_date",
            "sector",
            "source",
            "confidence_level",
            "auto_duplicate_flag",
            "duplicate_group",
            "duplicate_confidence",
            "duplicate_reason"
        ]]
        return "Potential duplicate comps detected using automatic duplicate scoring.", result

    if intent == "fitzrovia":
        if not can(user_role, "view_deals"):
            return "Access restricted.", "Your role does not have permission to query deal records."

        deal_results = deals_df[
            deals_df["address"].str.contains("Fitzrovia|Charlotte", case=False, na=False)
        ][[
            "address",
            "sector",
            "asking_price_m",
            "underwrite_irr",
            "stage"
        ]]

        if can(user_role, "view_comps"):
            comp_results = comps_df[
                comps_df["postcode"].str.contains("W1|W1T", case=False, na=False)
            ][[
                "address",
                "sector",
                "price_psf",
                "niy",
                "headline_rent_psf",
                "transaction_date"
            ]]
            result = pd.concat([deal_results, comp_results], ignore_index=True)
        else:
            result = deal_results

        return "Fitzrovia/W1 deal and comp evidence based on current role permissions.", result

    if intent == "call_this_week":
        if not can(user_role, "view_partners"):
            return "Access restricted.", "Your role does not have permission to query capital partner records."

        live_deals = deals_df[~deals_df["stage"].isin(["Passed", "Completed"])]
        rows = []

        for _, deal_row in live_deals.iterrows():
            matches = get_partner_matches(deal_row["id"]).head(3)

            for _, match in matches.iterrows():
                if match["status"] == "Active" and match["match_score"] >= 65:
                    rows.append({
                        "capital_partner": match["organisation_name"],
                        "key_contact": match["key_contact"],
                        "deal": deal_row["address"],
                        "deal_stage": deal_row["stage"],
                        "match_score": match["match_score"],
                        "reason": match["reasons"]
                    })

        result = pd.DataFrame(rows).sort_values("match_score", ascending=False)
        return "Capital partners to prioritise this week based on active status and live deal fit.", result

    if intent == "comps_for_selected_deal":
        if not can(user_role, "view_comps"):
            return "Access restricted.", "Your role does not have permission to query comps records."

        deal = deals_df[deals_df["id"] == selected_deal_id].iloc[0].to_dict()
        result = get_relevant_comps(deal)[[
            "id",
            "comp_type",
            "address",
            "postcode",
            "transaction_date",
            "sector",
            "size_sqft",
            "price_psf",
            "niy",
            "headline_rent_psf",
            "auto_duplicate_flag",
            "duplicate_confidence"
        ]]
        return f"Relevant comps for selected deal: {deal['address']}.", result

    if intent == "high_irr_deals":
        if not can(user_role, "view_deals"):
            return "Access restricted.", "Your role does not have permission to query deal records."

        threshold = extract_number(q, 15)
        result = deals_df[
            (deals_df["underwrite_irr"] >= threshold) &
            (~deals_df["stage"].isin(["Passed", "Completed"]))
        ][[
            "address",
            "sector",
            "asking_price_m",
            "underwrite_irr",
            "equity_requirement_m",
            "stage"
        ]]
        return f"Live deals with underwrite IRR at or above {threshold}%.", result

    if intent == "passed_deals":
        if not can(user_role, "view_deals"):
            return "Access restricted.", "Your role does not have permission to query deal records."

        result = deals_df[deals_df["stage"] == "Passed"][[
            "address",
            "sector",
            "asking_price_m",
            "underwrite_irr",
            "stage",
            "deal_summary"
        ]]
        return "Passed deals and summary rationale.", result

    return (
        "Unknown query intent.",
        "I can answer demo queries about: partners with IRR above 15, office deals, duplicate comps, Fitzrovia/W1 evidence, partners to call this week, comps for this deal, high IRR deals, and passed deals."
    )


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")

user_role = st.sidebar.selectbox(
    "Prototype user role",
    ["Investment Team", "Capital Markets", "Asset Manager", "Fund Administrator"]
)

st.sidebar.info(f"Current role: {user_role}")

with st.sidebar.expander("Role permissions"):
    st.json(ROLE_PERMISSIONS[user_role])

anthropic_api_key_input = st.sidebar.text_input(
    "Claude API key",
    type="password",
    help=(
        "Paste an Anthropic API key to enable live Claude outputs for this session. "
        "The key is not stored in the app or committed to the repository. "
        "For testing without a real API key, enter TEST_MODE to confirm the runtime key flow."
    )
)

if anthropic_api_key_input:
    st.sidebar.success("Claude API key provided for this session.")
elif os.getenv("ANTHROPIC_API_KEY"):
    st.sidebar.success("Claude API key loaded from environment variable.")
else:
    st.sidebar.warning("No Claude API key provided. AI buttons will show a setup message.")

selected_deal_id = st.sidebar.selectbox(
    "Select pipeline deal",
    deals_df["id"] + " — " + deals_df["address"]
).split(" — ")[0]

if can(user_role, "view_partners"):
    selected_partner_id = st.sidebar.selectbox(
        "Select capital partner",
        partners_df["id"] + " — " + partners_df["organisation_name"]
    ).split(" — ")[0]
else:
    selected_partner_id = None
    st.sidebar.warning("Partner selector hidden for this role.")


# -----------------------------
# Security banner
# -----------------------------
st.info(
    "Prototype security model: this demo uses dummy data persisted in a local SQLite database "
    "and a role selector to demonstrate access-control behaviour at the UI level. "
    "No real proprietary data is used, and API keys are handled via environment variables "
    "or an optional session-only sidebar input. "
    "A production implementation would replace this with secure authentication (e.g. SSO), "
    "fully enforced role-based access control, encrypted storage at rest and in transit, "
    "comprehensive audit logging, and enterprise-grade AI data governance to control retention "
    "and prevent use of sensitive data in external model training."
)


# -----------------------------
# KPI row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Capital partners", len(partners_df) if can(user_role, "view_partners") else "Restricted")
col2.metric("Pipeline deals", len(deals_df) if can(user_role, "view_deals") else "Restricted")
col3.metric("Comps", len(comps_df) if can(user_role, "view_comps") else "Restricted")
col4.metric(
    "Auto duplicate comps flagged",
    int(comps_df["auto_duplicate_flag"].sum()) if can(user_role, "view_comps") else "Restricted"
)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Deal Pipeline",
    "Capital Partners",
    "Comps",
    "AI Intelligence",
    "Natural Language Query",
    "Production Roadmap",
    "App Specification"
])


with tab1:
    st.subheader("Deal Pipeline")

    if can(user_role, "view_deals"):
        st.dataframe(deals_df, use_container_width=True)

        st.markdown("### Selected deal")
        selected_deal = deals_df[deals_df["id"] == selected_deal_id].iloc[0]
        st.json(selected_deal.to_dict())
    else:
        access_denied_message("deal pipeline records")


with tab2:
    st.subheader("Capital Partners")

    if can(user_role, "view_partners"):
        visible_partners_df = redact_partner_data(partners_df, user_role)
        st.dataframe(visible_partners_df, use_container_width=True)

        if selected_partner_id:
            st.markdown("### Selected partner")
            selected_partner = partners_df[partners_df["id"] == selected_partner_id].iloc[0].to_dict()
            st.json(redact_partner_data(pd.DataFrame([selected_partner]), user_role).iloc[0].to_dict())
    else:
        access_denied_message("capital partner records")


with tab3:
    st.subheader("Investment and Leasing Comps")

    if can(user_role, "view_comps"):
        st.dataframe(comps_df, use_container_width=True)

        st.markdown("### Automatic duplicate detection")
        st.caption(
            "Duplicates are detected through address similarity, postcode match, transaction date proximity, "
            "size variance and pricing/rent/yield similarity."
        )

        if duplicate_pairs_df.empty:
            st.success("No duplicate comp pairs detected.")
        else:
            st.dataframe(duplicate_pairs_df, use_container_width=True)

        st.markdown("### Flagged comp records")
        flagged_comps = comps_df[comps_df["auto_duplicate_flag"] == True]
        st.dataframe(flagged_comps, use_container_width=True)
    else:
        access_denied_message("investment and leasing comps")


with tab4:
    st.subheader("AI Intelligence")

    if not can(user_role, "view_deals"):
        access_denied_message("AI intelligence because deal access is required")
    else:
        selected_deal = deals_df[deals_df["id"] == selected_deal_id].iloc[0].to_dict()

        if can(user_role, "view_partners"):
            selected_partner = partners_df[partners_df["id"] == selected_partner_id].iloc[0].to_dict()
            matches_df = get_partner_matches(selected_deal_id)
        else:
            selected_partner = None
            matches_df = pd.DataFrame()

        if can(user_role, "view_comps"):
            relevant_comps_df = get_relevant_comps(selected_deal)
        else:
            relevant_comps_df = pd.DataFrame()

        safe_deal, safe_partner, safe_comps, safe_matches = redact_ai_context_for_role(
            role=user_role,
            selected_deal=selected_deal,
            selected_partner=selected_partner,
            relevant_comps=relevant_comps_df,
            matches=matches_df
        )

        st.markdown("### 1. Cross-database capital partner matching")

        if can(user_role, "view_partners"):
            st.write(f"Selected deal: **{selected_deal['address']}**")
            st.dataframe(
                redact_partner_data(matches_df[[
                    "organisation_name",
                    "partner_type",
                    "status",
                    "match_score",
                    "reasons",
                    "meeting_notes"
                ]], user_role),
                use_container_width=True
            )
        else:
            access_denied_message("capital partner matching results")

        st.markdown("### 2. Relevant comps for selected deal")

        if can(user_role, "view_comps"):
            st.dataframe(relevant_comps_df, use_container_width=True)
        else:
            access_denied_message("comps matched to the selected deal")

        with st.expander("Source data available to AI for this role"):
            st.markdown("#### Selected deal")
            st.json(safe_deal)

            st.markdown("#### Selected partner")
            if safe_partner:
                st.json(safe_partner)
            else:
                st.write("Restricted or not available for this role.")

            st.markdown("#### Relevant comps")
            if not safe_comps.empty:
                st.dataframe(safe_comps, use_container_width=True)
            else:
                st.write("Restricted or not available for this role.")

            st.markdown("#### Partner matches")
            if not safe_matches.empty:
                st.dataframe(safe_matches, use_container_width=True)
            else:
                st.write("Restricted or not available for this role.")

        st.markdown("### 3. AI-generated outputs")

        if not can(user_role, "generate_ai"):
            access_denied_message("AI-generated outputs")
        else:
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("Generate meeting brief"):
                    with st.spinner("Generating meeting brief..."):
                        if not selected_partner_id:
                            brief = "Access restricted: no partner available for this role."
                        else:
                            brief = generate_meeting_brief(
                                selected_partner_id,
                                user_role,
                                anthropic_api_key_input
                            )

                        st.markdown(brief)

                        if can(user_role, "download_reports"):
                            st.download_button(
                                label="Download meeting brief",
                                data=brief,
                                file_name=f"meeting_brief_{selected_partner_id or 'restricted'}.md",
                                mime="text/markdown"
                            )

            with col_b:
                if st.button("Generate comps report"):
                    with st.spinner("Generating comps report..."):
                        report = generate_comps_report(
                            selected_deal_id,
                            user_role,
                            anthropic_api_key_input
                        )
                        st.markdown(report)

                        if can(user_role, "download_reports"):
                            st.download_button(
                                label="Download comps report",
                                data=report,
                                file_name=f"comps_report_{selected_deal_id}.md",
                                mime="text/markdown"
                            )

            with col_c:
                if st.button("Generate investment analysis"):
                    with st.spinner("Generating investment analysis..."):
                        analysis = generate_investment_analysis(
                            selected_deal_id,
                            user_role,
                            anthropic_api_key_input
                        )
                        st.markdown(analysis)

                        if can(user_role, "download_reports"):
                            st.download_button(
                                label="Download investment analysis",
                                data=analysis,
                                file_name=f"investment_analysis_{selected_deal_id}.md",
                                mime="text/markdown"
                            )

        st.markdown("### Architecture note")
        st.info(
            "This prototype uses deterministic filtering for deal/partner suitability and Claude-ready "
            "source-grounded prompts for narrative outputs. The role selector demonstrates access-control "
            "behaviour, although production authentication would be handled through SSO."
        )

        st.warning(
            "AI outputs are generated from structured internal data and should be reviewed by a human "
            "before being used externally."
        )


with tab5:
    st.subheader("Natural Language Query")

    st.markdown("### Supported query examples")
    st.markdown("""
- Which partners target IRR above 15?
- Show me office deals
- Show duplicate comps
- What have we seen in Fitzrovia?
- Which capital partners should we call this week?
- Build comps for this deal
- Show live deals with IRR above 18
- Show passed deals
""")

    query = st.text_input(
        "Ask a question across the datasets",
        placeholder="Example: Which capital partners should we call this week?"
    )

    if query:
        interpretation, result = run_structured_query(query, selected_deal_id, user_role)

        st.markdown("### Query interpretation")
        st.info(interpretation)

        if isinstance(result, pd.DataFrame):
            if result.empty:
                st.warning("No matching records found.")
            else:
                st.dataframe(result, use_container_width=True)

                if can(user_role, "generate_ai"):
                    if st.button("Summarise query result with AI"):
                        prompt = f"""
You are an assistant for a real estate investment team.

Task:
Summarise the structured query result below.

Rules:
- Use only the provided query result.
- Do not invent facts or wider market context.
- If the result is empty, say no matching records were found.
- State how you interpreted the user's query.
- Keep the answer structured, concise and actionable.

User query:
{query}

Query interpretation:
{interpretation}

Query result:
{result.to_json(orient="records", indent=2)}

Output:
1. Query interpreted as
2. Key results
3. Commercial relevance
4. Suggested next action
"""
                        with st.spinner("Summarising..."):
                            summary = call_ai(prompt, anthropic_api_key_input)
                            st.markdown(summary)

                            if can(user_role, "download_reports"):
                                st.download_button(
                                    label="Download query summary",
                                    data=summary,
                                    file_name="query_summary.md",
                                    mime="text/markdown"
                                )
                else:
                    st.warning("Your role does not have permission to generate AI summaries.")
        else:
            st.warning(result)


with tab6:
    st.subheader("Production Roadmap")

    st.markdown(
        """
### Current prototype security

The submitted app uses dummy data persisted in a local SQLite database and a prototype role selector. It demonstrates role-based access behaviour at the UI level, but does not implement production authentication or database-level authorisation.

Implemented in the prototype:

- Dummy data is stored in a local SQLite database
- No real proprietary deal, investor or transaction data is used
- No API keys are committed to the codebase
- API keys are read from environment variables or entered at runtime for the current Streamlit session
- Mock AI mode can be enabled if the app needs to run without sending data to external AI APIs
- Role-based visibility is demonstrated in the UI
- AI context is reduced or restricted based on selected role
- Report downloads can be disabled for lower-permission roles

### Deployment and runtime API key handling

The prototype is intended to be hosted so reviewers can access the app without setting up the project locally.

To allow live AI testing without hardcoding credentials:

- Reviewers can enter an optional Claude API key in the sidebar
- The key is used only for the current session
- The key is not committed to the repository
- The app also supports `ANTHROPIC_API_KEY` as an environment variable
- If no key is provided, the app returns a setup message rather than failing

This is a prototype convenience. In production, user-entered API keys would be removed and model calls would be routed through a secure backend AI service.

### Production authentication and access control

Production access would be enforced through:

- Single sign-on via Microsoft Entra ID or Google Workspace
- Role-Based Access Control [RBAC]
- Database-level permissions
- Route/API-level permission checks
- Separate read/write permissions by team and asset/deal sensitivity
- Privileged admin controls for managing access

### Data storage

The current prototype uses SQLite because it provides lightweight persistence without requiring external infrastructure.

In production, this would be replaced by PostgreSQL with:

- Normalised tables for partners, contacts, deals, notes, comps and AI reports
- Foreign key relationships between records
- Indexed fields for sector, geography, IRR, stage and transaction date
- Proper migration management
- Backup and recovery procedures
- Database-level access controls

### Third-party AI API handling

Claude would be accessed through a secure backend service, not directly from the frontend.

Controls would include:

- No API keys in client-side code
- Environment-secret management
- Data minimisation before prompt construction
- Redaction of unnecessary personal/confidential fields
- Enterprise API settings preventing training on customer data
- Defined prompt/response retention policy
- Logging of source record IDs without storing unnecessary raw sensitive content

### Encryption

Production data security would include:

- Encryption at rest for PostgreSQL and document storage
- TLS/HTTPS for data in transit
- Secret management for API keys and database credentials
- Optional field-level encryption for highly sensitive investor or deal data

### Audit and governance

The production system would maintain an audit trail for:

- User logins
- Record reads and writes
- AI report generation
- Source records used in AI prompts
- Human approvals for AI-suggested updates
- Email/voice ingestion decisions

### Voice ingestion

Voice notes would be handled through:

1. Speech-to-text transcription
2. LLM classification of note type: meeting note, deal update or comp entry
3. Entity matching to capital partner, deal or comp record
4. Human review screen showing the proposed database update
5. Approved write to the database with audit log

### Email ingestion

Incoming emails would follow:

1. Secure mailbox connection
2. Email parsing and attachment extraction
3. Classification into meeting note, deal update, underwrite, or market comp
4. Field extraction into structured schema
5. Human approval before database write
6. Audit trail showing source email, suggested update and approving user

### Production architecture

The Streamlit prototype would become:

- React front end
- FastAPI backend
- PostgreSQL database
- Secure document storage
- Claude-powered AI service layer
- Retrieval-Augmented Generation [RAG] over structured data and documents
- Human review workflow for AI-suggested database updates
"""
    )

with tab7:
    st.title("Integrated Intelligence Platform — Specification")

    st.info(
        "This page summarises what the prototype does, why it was built this way, "
        "how AI is used, and how the system would be taken into production."
    )

    # -----------------------------
    # Executive Summary
    # -----------------------------
    st.markdown("## Executive Summary")

    st.markdown(
        """
This platform brings together three core datasets — **capital partners**, **live deals**, 
and **market comparables** — into one integrated system.

It helps users:

- identify the most relevant capital partners for each deal
- generate meeting briefs ahead of investor calls
- benchmark deals against comparable market evidence
- query deal, investor and comps data in plain English

The aim is to reduce manual cross-referencing across spreadsheets, notes and separate tools, 
while improving the speed and consistency of investment decision-making.
"""
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Core datasets", "3")

    with col2:
        st.metric("AI outputs", "4")

    with col3:
        st.metric("Query intents", "8")

    with col4:
        st.metric("Prototype score", "9 / 10")

    st.divider()

    # -----------------------------
    # Architecture
    # -----------------------------
    st.markdown("## 1. Architecture Overview")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(
            """
### What the system does

The platform connects three previously separate datasets:

- **Deals** matched with **capital partners**
- **Deals** benchmarked against **market comps**
- **Users** able to query everything in one place

AI is used to **summarise and explain**, not to make calculations or final decisions.
"""
        )

    with col_b:
        st.code(
            """
User Interface (Streamlit)
        ↓
Application Logic (Python)
        ↓
SQLite Database
        ↓
AI Layer (Claude / TEST_MODE)
""",
            language="text"
        )

    st.success(
        "Key design principle: rule-based logic handles matching, filtering and scoring; "
        "AI is used for written outputs such as meeting briefs, reports and summaries."
    )

    st.divider()

    # -----------------------------
    # Tech Stack
    # -----------------------------
    st.markdown("## 2. Tech Stack Rationale")

    tech_stack = pd.DataFrame(
        [
            {
                "Component": "Streamlit",
                "Used for": "Frontend and application interface",
                "Why chosen": "Fast to build, easy to deploy, suitable for internal data tools",
            },
            {
                "Component": "SQLite",
                "Used for": "Persistent prototype database",
                "Why chosen": "Lightweight, local, no infrastructure required",
            },
            {
                "Component": "pandas",
                "Used for": "Filtering, scoring and data preparation",
                "Why chosen": "Fast manipulation of structured datasets",
            },
            {
                "Component": "Claude",
                "Used for": "AI-generated briefs, reports and summaries",
                "Why chosen": "Strong at structured reasoning and written synthesis",
            },
        ]
    )

    st.dataframe(tech_stack, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------
    # Data Model
    # -----------------------------
    st.markdown("## 3. Data Model")

    data_col1, data_col2, data_col3 = st.columns(3)

    with data_col1:
        st.markdown(
            """
### Capital Partners

Stores:

- organisation and contact details
- investor mandate
- ticket size
- target returns
- preferred sectors
- meeting notes
- engagement status

**Purpose:** identifies which investors are relevant for each deal.
"""
        )

    with data_col2:
        st.markdown(
            """
### Deal Pipeline

Stores:

- property details
- pricing
- initial yield
- underwrite IRR
- equity requirement
- stage
- key dates
- deal summary

**Purpose:** acts as the central dataset for matching, reporting and analysis.
"""
        )

    with data_col3:
        st.markdown(
            """
### Market Comps

Stores:

- investment transactions
- leasing evidence
- pricing
- yields
- rents
- source
- confidence level
- duplicate flags

**Purpose:** provides market evidence for investment decisions.
"""
        )

    with st.expander("Derived data created by the app"):
        st.markdown(
            """
- Capital partner match scores  
- Duplicate comp scores and groupings  
- Structured query results  
- AI-ready source data payloads  
- Exportable meeting briefs, comps reports and investment analysis outputs  
"""
        )

    st.divider()

    # -----------------------------
    # AI Integration
    # -----------------------------
    st.markdown("## 4. AI Integration")

    ai_outputs = pd.DataFrame(
        [
            {
                "AI feature": "Meeting Brief",
                "Input data": "Capital partner + matched deals",
                "Output": "Investor profile, deals to raise, questions and follow-up actions",
                "Business value": "Reduces meeting preparation time",
            },
            {
                "AI feature": "Comps Report",
                "Input data": "Selected deal + relevant comps",
                "Output": "External-ready market evidence report",
                "Business value": "Improves consistency and speed of reporting",
            },
            {
                "AI feature": "Investment Analysis",
                "Input data": "Deal + comps + partner matches",
                "Output": "First-pass deal assessment, risks and diligence questions",
                "Business value": "Supports faster early-stage decision-making",
            },
            {
                "AI feature": "Query Summary",
                "Input data": "Structured query result",
                "Output": "Concise business summary",
                "Business value": "Turns raw results into actionable insight",
            },
        ]
    )

    st.dataframe(ai_outputs, use_container_width=True, hide_index=True)

    st.warning(
        "AI is deliberately not used for calculations, scoring or permission logic. "
        "Those are handled by deterministic code first, then AI explains the result."
    )

    with st.expander("Prompting approach"):
        st.markdown(
            """
The prompts are designed to keep outputs controlled and grounded:

- use only the provided data
- do not invent facts, names, dates, metrics or market evidence
- state when information is not available
- separate facts from judgement calls
- return structured, business-readable outputs

This reduces hallucination risk and makes outputs easier to review.
"""
        )

    st.divider()

    # -----------------------------
    # Deployment / API Key Handling
    # -----------------------------
    st.markdown("## 5. Deployment and API Key Approach")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
### Hosted prototype

The app is designed to be hosted so reviewers can access it through a link without local setup.

This avoids requiring assessors to:

- install dependencies
- clone the repository
- configure local environment variables
- run Streamlit manually
"""
        )

    with col_right:
        st.markdown(
            """
### Runtime Claude API key

The app allows a reviewer to enter a Claude API key in the sidebar.

The key is:

- used only for the current session
- not stored in the database
- not committed to the repository
- optional for testing
"""
        )

    st.info(
        "Entering TEST_MODE in the Claude API key field confirms that the runtime key routing works "
        "without making an external API call."
    )

    st.divider()

    # -----------------------------
    # Security
    # -----------------------------
    st.markdown("## 6. Data Security Approach")

    sec1, sec2 = st.columns(2)

    with sec1:
        st.markdown(
            """
### Implemented in the prototype

- Dummy data only
- SQLite persistence
- No API keys stored in code
- Runtime API key input
- Prototype role-based visibility
- AI context restricted by selected role
- Report downloads controlled by role
"""
        )

    with sec2:
        st.markdown(
            """
### Prototype limitations

- No real authentication
- No backend-level enforcement
- SQLite is not encrypted
- No production audit logging
- No enterprise secret management
"""
        )

    with st.expander("Production security model"):
        st.markdown(
            """
A production version would include:

- Single sign-on through Microsoft Entra ID or Google Workspace
- Role-based access control enforced at API and database level
- encrypted storage at rest and in transit
- backend-managed Claude API access
- audit logs for user actions and AI calls
- data minimisation and redaction before AI processing
- enterprise AI data-retention controls
"""
        )

    st.divider()

    # -----------------------------
    # Task 1 Self-Assessment
    # -----------------------------
    st.markdown("## 7. Task 1 Self-Assessment")

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        st.metric("Overall assessment", "9 / 10")

    with score_col2:
        st.metric("Strongest area", "AI integration")

    with score_col3:
        st.metric("Main limitation", "Prototype scope")

    assessment_df = pd.DataFrame(
        [
            {
                "Area": "Technical Build",
                "What was done well": "Working hosted app with integrated data, SQLite persistence and exportable outputs",
                "With more time": "Refactor into React + FastAPI with automated tests",
            },
            {
                "Area": "AI Integration",
                "What was done well": "AI used for summaries and reports, not calculations",
                "With more time": "Add document-aware retrieval and specialist AI workflows",
            },
            {
                "Area": "Cross-Database Intelligence",
                "What was done well": "Deals matched to capital partners and comps",
                "With more time": "Add historical feedback and configurable scoring weights",
            },
            {
                "Area": "Natural Language Querying",
                "What was done well": "Structured query routing with clear supported examples",
                "With more time": "Move to validated natural-language-to-SQL",
            },
            {
                "Area": "Comps and Data Cleaning",
                "What was done well": "Automatic duplicate detection and comp report generation",
                "With more time": "Add geographic radius, size bands and confidence weighting",
            },
            {
                "Area": "Security",
                "What was done well": "Role-based visibility, API key handling and restricted AI context",
                "With more time": "Implement SSO, backend RBAC, encryption and audit logs",
            },
            {
                "Area": "Product UX",
                "What was done well": "Clear tabs, KPIs, buttons, downloads and source transparency",
                "With more time": "Add dashboards, alerts, saved queries and polished PDF exports",
            },
        ]
    )

    st.dataframe(assessment_df, use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------
# What I Would Do With More Time (AI-Lead Focused)
# -----------------------------
st.markdown("## 8. AI and Platform Roadmap (What I Would Do If Given More Time)")

with st.expander("1. Introduce a Structured AI Layer", expanded=True):
    st.markdown(
        """
Move from direct prompt calls to a dedicated AI service layer within the application.

This would include:

- centralised prompt management
- standardised input and output schemas
- logging of all AI interactions
- versioning of prompts and models

**Benefit:**  
Makes AI behaviour consistent, testable and auditable across the platform, rather than ad hoc.
"""
    )

with st.expander("2. Implement Document-Aware AI (RAG)"):
    st.markdown(
        """
Extend the system beyond structured tables to include unstructured data:

- underwrites  
- IC papers  
- emails  
- meeting notes  

AI responses would be grounded in retrieved internal data, not just prompts.

**Benefit:**  
Enables deeper insight such as:
- “Why did we pass on this deal?”
- “What concerns did this investor raise previously?”
"""
    )

with st.expander("3. Introduce Task-Specific AI Workflows"):
    st.markdown(
        """
Replace generic AI calls with structured, task-specific workflows:

- Investment analysis workflow  
- Capital raising workflow  
- Comps benchmarking workflow  

Each workflow would:

- take structured inputs  
- perform multi-step reasoning  
- return structured outputs  

**Benefit:**  
Aligns AI outputs with real analyst workflows, improving reliability and usability.
"""
    )

with st.expander("4. AI Evaluation and Feedback Loops"):
    st.markdown(
        """
Introduce systems to measure and improve AI performance over time:

- user feedback (useful / not useful)
- tracking which outputs are used in real workflows
- evaluation datasets for testing prompts
- automated prompt regression testing

**Benefit:**  
Turns AI from a static feature into a continuously improving system.
"""
    )

with st.expander("5. AI Security and Governance"):
    st.markdown(
        """
Extend the security model specifically for AI usage:

- prompt injection protection
- strict data scoping before model calls
- redaction of sensitive fields (investor names, pricing)
- no retention of sensitive prompts in external systems
- audit logging of all AI requests and responses
- human-in-the-loop approval for critical outputs

**Benefit:**  
Prevents data leakage and ensures AI outputs are controlled, traceable and safe.
"""
    )

with st.expander("6. Role-Aware AI Outputs"):
    st.markdown(
        """
Adapt AI responses based on user role:

- capital markets → investor-focused outputs  
- asset management → operational insights  
- fund administration → structured reporting  

**Benefit:**  
Ensures outputs are relevant and prevents unnecessary exposure of sensitive information.
"""
    )

with st.expander("7. Production AI Architecture"):
    st.markdown(
        """
Move from direct API calls to a production-grade AI architecture:

- backend AI service layer
- managed secrets (no user-entered keys)
- request validation and rate limiting
- monitoring and alerting
- centralised logging of AI usage

**Benefit:**  
Aligns with enterprise AI deployment standards and improves scalability, reliability and security.
"""
    )

    st.divider()

    # -----------------------------
    # Final Summary
    # -----------------------------
    st.markdown("## Final Summary")

    st.success(
        "This prototype demonstrates how fragmented deal, investor and market data can be unified "
        "into a single intelligence platform. It is usable, explainable and commercially relevant, "
        "while providing a clear path to a secure production-ready system."
    )



# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown("""
### Production extension path

This prototype demonstrates the core intelligence loop:

1. Structured deal, partner and comp data
2. Deterministic filtering and scoring
3. Automatic duplicate detection
4. Structured natural language query routing
5. Prototype role-based access behaviour
6. Source-grounded AI outputs
7. Runtime Claude API key input for hosted review
8. Human-review-first workflow for production use

The main production build would separate the prototype into a front end, backend API,
database, secure document store and auditable AI service layer.
""")