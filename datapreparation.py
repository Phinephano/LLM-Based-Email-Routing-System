
"""
Data Preparation Module
=======================

This module prepares the Customer Support Ticket dataset
for transformer-based email routing experiments.
"""

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


REQUIRED_DEPARTMENTS = [
    "Technical Support",
    "Customer Service",
    "Billing and Payments",
    "Sales and Pre-Sales",
    "General Inquiry",
]


def load_and_prepare_data(seed=42):
    """
    Load and preprocess the Customer Support Tickets dataset.

    Returns:
        train_ds, val_ds, test_ds, label_list, label2id, id2label
    """

    dataset = load_dataset(
        "Tobi-Bueck/customer-support-tickets",
        split="train"
    )

    df = dataset.to_pandas().copy()

    # Keep English tickets only
    df = df[df["language"] == "en"].copy()

    # Keep only the five target departments
    df = df[df["queue"].isin(REQUIRED_DEPARTMENTS)].copy()

    # Handle missing text fields
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine subject and body
    df["text"] = (
        "Subject: " + df["subject"].astype(str)
        + "\\n\\nBody: " + df["body"].astype(str)
    )

    # Keep only required columns
    df = df[["text", "queue"]].copy()

    # Encode labels
    label_list = sorted(df["queue"].unique())

    label2id = {
        label: idx
        for idx, label in enumerate(label_list)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    df["label"] = df["queue"].map(label2id)

    # Split dataset
    train_df, test_df = train_test_split(
        df[["text", "label"]],
        test_size=0.20,
        random_state=seed,
        stratify=df["label"]
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        random_state=seed,
        stratify=train_df["label"]
    )

    train_ds = Dataset.from_pandas(
        train_df.reset_index(drop=True)
    )

    val_ds = Dataset.from_pandas(
        val_df.reset_index(drop=True)
    )

    test_ds = Dataset.from_pandas(
        test_df.reset_index(drop=True)
    )

    return (
        train_ds,
        val_ds,
        test_ds,
        label_list,
        label2id,
        id2label,
    )
