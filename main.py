import os
import re
import io
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import gradio as gr
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import tempfile
def narrate_text(text: str) -> str:
    """Convert chatbot output to speech and return audio filepath"""
    if not text or text.strip() == "":
        return None
    tts = gTTS(text=text, lang="en")
    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(tmp_path)
    return tmp_path
model_id = "ibm-granite/granite-3.3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)
model.eval()
print("✅ Granite model loaded!")
class UserProfile(BaseModel):
    persona: str = Field(description="student or professional")
    location: Optional[str] = None
    currency: str = "USD"
    monthly_income: Optional[float] = None

@dataclass
class BudgetSummary:
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate: float
    by_category: Dict[str, float]
    flags: List[str]

DEFAULT_EXPENSE_CATS = [
    "Housing", "Food", "Transport", "Utilities", "Insurance", "Healthcare",
    "Debt", "Education", "Entertainment", "Shopping", "Savings/Investments", "Other",
]

def currency_prefix(cur: str) -> str:
    return "$" if cur.upper() == "USD" else (cur.upper() + " ")
def normalize_category(raw: str) -> str:
    cat = str(raw).strip().lower()
    if re.search(r"sav|invest", cat):
        return "Savings/Investments"
    if re.search(r"house|rent|mortg", cat):
        return "Housing"
    if re.search(r"food|grocer|dining|restaurant", cat):
        return "Food"
    if re.search(r"trans|travel|commut|uber|lyft|ride|fuel|gas", cat):
        return "Transport"
    if re.search(r"util|electric|water|internet|wifi|phone|gas bill", cat):
        return "Utilities"
    if re.search(r"insur", cat):
        return "Insurance"
    if re.search(r"health|med|doctor|pharma|dental|vision", cat):
        return "Healthcare"
    if re.search(r"debt|loan|emi|mortgage payment|credit card", cat):
        return "Debt"
    if re.search(r"educat|tuition|course|exam|school|college", cat):
        return "Education"
    if re.search(r"entertain|movie|fun|gaming|music|concert", cat):
        return "Entertainment"
    if re.search(r"shop|cloth|apparel|amazon|flipkart", cat):
        return "Shopping"
    return "Other"

def summarize_budget(income: float, expenses: Dict[str, float]) -> BudgetSummary:
    total_expenses = sum(expenses.values())
    net_savings = income - total_expenses
    savings_rate = (net_savings / income * 100) if income > 0 else 0.0

    flags: List[str] = []
    needs = sum(expenses.get(k, 0.0) for k in ["Housing", "Food", "Transport", "Utilities", "Healthcare", "Insurance"]) / max(income, 1)
    wants = sum(expenses.get(k, 0.0) for k in ["Entertainment", "Shopping"]) / max(income, 1)
    savings_part = (expenses.get("Savings/Investments", 0.0) + max(net_savings, 0.0)) / max(income, 1)

    if needs > 0.5:
        flags.append("Needs spend above 50% guideline")
    if wants > 0.3:
        flags.append("Wants spend above 30% guideline")
    if savings_part < 0.2:
        flags.append("Savings below 20% guideline")

    debt_ratio = expenses.get("Debt", 0.0) / max(income, 1)
    if debt_ratio > 0.36:
        flags.append("Debt payments above 36% of income (DTI warning)")

    return BudgetSummary(
        total_income=income,
        total_expenses=total_expenses,
        net_savings=net_savings,
        savings_rate=savings_rate,
        by_category=expenses,
        flags=flags,
    )

def build_system_prompt(profile: UserProfile) -> str:
    income_note = (
        f" The user's monthly salary is {profile.currency} {profile.monthly_income:,.0f}."
        if (profile.monthly_income is not None and profile.monthly_income > 0) else ""
    )
    currency_note = f" Always express money amounts in {profile.currency}."
    location_note = f" The user is located in {profile.location}." if profile.location else ""

    if profile.persona == "student":
        return (
            "You are a friendly personal finance tutor for a college student. "
            "Use simple language, short sentences, and examples. Avoid jargon. "
            "Be encouraging and focus on fundamentals. Always include a quick TL;DR."
            + income_note + currency_note + location_note
        )
    else:
        return (
            "You are a professional financial assistant for a working adult. "
            "Provide concise, structured, and actionable guidance with light caveats. "
            "Use bullet points and include a brief summary line."
            + income_note + currency_note + location_note
        )

def granite_generate(prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def default_expense_df() -> pd.DataFrame:
    return pd.DataFrame({"category": DEFAULT_EXPENSE_CATS, "amount": [0.0]*len(DEFAULT_EXPENSE_CATS)})

def load_table_from_file(file_obj) -> Tuple[float, pd.DataFrame]:
    """
    Returns (income, dataframe with columns: category, amount)
    """
    if file_obj is None:
        return 0.0, default_expense_df()

    path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = (os.path.splitext(path)[1] or "").lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    else:
        return 0.0, default_expense_df()

    # Normalize columns
    df.columns = [str(c).strip() for c in df.columns]

    # Case 1: long format with 'category' and 'amount'
    has_cat_amt = set([c.lower() for c in df.columns]) >= {"category", "amount"}

    if has_cat_amt:
        df2 = df.rename(columns={c: c.lower() for c in df.columns})[["category", "amount"]]
        # Normalize categories via regex mapping
        df2["category"] = df2["category"].apply(normalize_category)
        grouped = df2.groupby("category", as_index=False)["amount"].sum()
        # Ensure all defaults present
        full = default_expense_df()
        for _, row in grouped.iterrows():
            full.loc[full["category"] == row["category"], "amount"] = float(row["amount"])
        # Check for income column too (optional in long format)
        income = 0.0
        for c in df.columns:
            cl = c.lower()
            if cl in ("income", "salary", "monthly income"):
                try:
                    income = float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
                except:
                    pass
        return income, full

    income = 0.0
    values = {cat: 0.0 for cat in DEFAULT_EXPENSE_CATS}
    for c in df.columns:
        lc = c.lower()
        if lc in ("income", "salary", "monthly income"):
            try:
                income = float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
                print(f"Found income column '{c}': {income}")
            except Exception as e:
                print(f"Error parsing income column '{c}': {e}")
                pass
            continue
        mapped = normalize_category(c)
        try:
            values[mapped] += float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
        except:
            pass

    full = default_expense_df()
    for cat in DEFAULT_EXPENSE_CATS:
        full.loc[full["category"] == cat, "amount"] = float(values.get(cat, 0.0))
    return income, full


def table_to_text(df: pd.DataFrame, currency: str) -> str:
    # Pretty text table for LLM grounding
    buf = io.StringIO()
    df_out = df.copy()
    df_out["amount"] = df_out["amount"].fillna(0.0).map(lambda x: f"{x:,.2f}")
    buf.write(f"(Amounts in {currency})\n")
    buf.write(df_out.to_string(index=False))
    return buf.getvalue()

def run_chat(profile: UserProfile, user_input: str, table_df: Optional[pd.DataFrame]) -> Tuple[str, Optional[BudgetSummary]]:
    expenses: Dict[str, float] = {c: 0.0 for c in DEFAULT_EXPENSE_CATS}
    income = float(profile.monthly_income or 0.0)

    if table_df is not None and len(table_df) > 0:
        for _, row in table_df.iterrows():
            cat = normalize_category(row.get("category", "Other"))
            try:
                amt = float(row.get("amount", 0.0) or 0.0)
            except:
                amt = 0.0
            expenses[cat] = expenses.get(cat, 0.0) + amt

    summary = None
    if income > 0 and any(v > 0 for v in expenses.values()):
        summary = summarize_budget(income, expenses)

    system_prompt = build_system_prompt(profile)
    table_text = ""
    if table_df is not None and len(table_df) > 0:
        table_text = "\n\nHere is the uploaded/edited expense data:\n" + table_to_text(table_df, profile.currency)

    final_prompt = (
        system_prompt
        + "\n\n"
        + "User: "
        + (user_input or "Give me a budget summary and tips.")
        + table_text
        + "\nAssistant:"
    )

    model_text = granite_generate(final_prompt)
    return model_text, summary

CSS = """
#metrics span.value { font-weight:700; }
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# 💬 Granite Personal Finance Chatbot\n_Not financial advice. For education only._")
    with gr.Row():
     audio_out = gr.Audio(label="Narration", type="filepath")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(label="Chat", value=[])
            user_msg = gr.Textbox(label="Ask about savings, taxes, investments, or request a budget summary:")
            send_btn = gr.Button("Send") # send_btn defined here

        with gr.Column(scale=2):
            gr.Markdown("### Profile")
            persona = gr.Radio(["student", "professional"], value="student", label="Persona")
            currency = gr.Dropdown(["USD", "EUR", "INR", "GBP", "JPY", "AUD"], value="USD", label="Currency")
            location = gr.Textbox(value="", label="Location (optional)")
            monthly_income = gr.Number(label="Monthly income", value=0.0, precision=2)

            gr.Markdown("### Expenses")
            file_upload = gr.File(label="Upload CSV/XLSX (either 'category,amount' long format, or wide format + optional income)", type="filepath")

            table = gr.Dataframe(
                headers=["category", "amount"],
                value=default_expense_df(),
                datatype=["str", "number"],
                row_count=(len(DEFAULT_EXPENSE_CATS), "dynamic"),
                col_count=(2, "fixed"),
                wrap=True,
                label="Edit your expenses (category, amount)"
            )

            load_btn = gr.Button("Load file into table")

            gr.Markdown("### Budget Summary")
            income_md = gr.Markdown("")
            expenses_md = gr.Markdown("")
            netsave_md = gr.Markdown("")
            saverate_md = gr.Markdown("")
            flags_md = gr.Markdown("")
            chart_plot = gr.Plot(label="By Category")
    def handle_load(file_path):
        income, df = load_table_from_file(file_path)
        return df, float(income)

    load_btn.click(
        handle_load,
        inputs=[file_upload],
        outputs=[table, monthly_income]
    )

    def on_send(user_msg_txt, history, persona_v, currency_v, location_v, monthly_income_v, table_df):
        prof = UserProfile(
            persona=persona_v,
            currency=currency_v,
            monthly_income=float(monthly_income_v or 0.0),
            location=(location_v or "").strip() or None
        )


        df = pd.DataFrame(table_df, columns=["category", "amount"]).fillna({"category":"Other", "amount":0.0})

        reply, summary = run_chat(prof, user_msg_txt, df)

        # Narration (convert reply to speech)
        audio_file = narrate_text(reply)

        # Prepare summary outputs
        cur_pref = currency_prefix(prof.currency)
        income_txt = f"**Income:** {cur_pref}{(prof.monthly_income or 0):,.0f}"
        if summary:
            expenses_txt = f"**Expenses:** {cur_pref}{summary.total_expenses:,.0f}"
            netsave_txt = f"**Net Savings:** {cur_pref}{summary.net_savings:,.0f}"
            saverate_txt = f"**Savings Rate:** {summary.savings_rate:.1f}%"
            if summary.flags:
                flags_txt = "**Warnings:**\n" + "\n".join([f"- {f}" for f in summary.flags])
            else:
                flags_txt = "✅ Budget looks healthy!"

            # Plot by category
            fig = plt.figure()
            cats = list(summary.by_category.keys())
            vals = [summary.by_category[c] for c in cats]
            plt.bar(cats, vals)
            plt.xticks(rotation=60, ha="right")
            plt.ylabel("Amount")
            plt.title("Expenses by Category")
        else:
            expenses_txt = "**Expenses:** —"
            netsave_txt = "**Net Savings:** —"
            saverate_txt = "**Savings Rate:** —"
            flags_txt = "_Provide income and some expenses to see detailed budget analysis._"
            fig = plt.figure()
            plt.text(0.5, 0.5, "No expense data", ha="center", va="center")
            plt.axis("off")

        history = history + [(user_msg_txt, reply)]
        return history, "", income_txt, expenses_txt, netsave_txt, saverate_txt, flags_txt, fig, audio_file


    send_btn.click(
        on_send,
        inputs=[user_msg, chatbot_ui, persona, currency, location, monthly_income, table],
        outputs=[chatbot_ui, user_msg, income_md, expenses_md, netsave_md, saverate_md, flags_md, chart_plot, audio_out]
    )


demo.launch(share=True)
