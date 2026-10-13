import pandas as pd
from app.database import sessionLocal, Base, engine
from app.models import Message


def load_inline():
    data = {
        "label": ["ham"] * 12 + ["spam"] * 12,
        "message": [
            "Hey, are we still on for lunch tomorrow?",
            "Can you pick up some milk on your way home?",
            "Meeting moved to 3pm, see you there.",
            "Thanks for the update, I'll review it tonight.",
            "Happy birthday! Hope you have a great day.",
            "Did you finish the report? Need it by 5.",
            "Call me when you get a chance.",
            "The project deadline is next Friday.",
            "Let's catch up this weekend.",
            "Please send me the notes from class.",
            "Are you free tonight for a quick call?",
            "Sent the invoice this morning, check your inbox.",
            "WINNER!! You've won a FREE iPhone! Click http://bit.ly/xyz NOW!",
            "URGENT: Your account has been suspended. Verify at http://fake-bank.com",
            "Congratulations! Claim your $1000 prize. Text WIN to 80000",
            "FREE entry in a weekly competition to win FA Cup tickets!",
            "You have 1 new voicemail. Call 09061701461 to listen. Cost 1.50/min",
            "Get cheap loans now! No credit check required. Apply today!",
            "Hot singles in your area want to chat! Click here!",
            "Your mobile number has won 2000 pounds! Call 09050000336 to claim",
            "LIMITED OFFER: 50% off all products. Buy now at www.deals.com",
            "Lowest price for Viagra! Discreet shipping. Order today.",
            "FREE MONEY! No strings attached! Reply YES now!",
            "Urgent reply needed: claim your unclaimed funds today!",
        ],
    }
    return pd.DataFrame(data)


def main():
    Base.metadata.create_all(bind=engine)

    try:
        df = pd.read_csv("SMSSpamCollection.csv", sep="\t",
                         names=["label", "message"], encoding="latin-1")
        print(f"Loaded real dataset: {len(df)} rows")
    except FileNotFoundError:
        df = load_inline()
        print(f"Using inline sample: {len(df)} rows")

    db = sessionLocal()
    try:
        # Wipe existing seed rows to avoid duplicates on rerun
        db.query(Message).filter(Message.source == "seed").delete()
        db.commit()

        seen = set()
        inserted = 0
        for _, row in df.iterrows():
            key = (row["message"], row["label"])
            if key in seen:
                continue
            seen.add(key)
            db.add(Message(content=row["message"], label=row["label"], source="seed"))
            inserted += 1
        db.commit()
        print(f"Inserted {inserted} rows into `messages`.")
    finally:
        db.close()


if __name__ == "__main__":
    main()