import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Adi016/distilBertModel")
model = AutoModelForSequenceClassification.from_pretrained("Adi016/distilBertModel")

def classify(email) :

    promptForIntentclassification = """
    You are an AI who is an expert email analyzer and classification system specializing in real estate and legal documentation.
    Your task is to accurately classify incoming emails into one of 8 predefined business intents. You must also identify emails that
    contain mixed intents or fall outside these 8 categories.

    **Here are the 8 predefined business intents:**

    0.  **Intent_Amendment_Abstraction**: Emails requesting the extraction of new terms or highlighting changes introduced by a lease amendment compared to the original lease.
    1.  **Intent_Clause_Protect**: Emails requesting a review of lease clauses to detect potentially risky, missing, or unfavorable terms (e.g., subletting rights, break clauses, indemnity, assignment terms, unreasonable liabilities, compliance issues).
    2.  **Intent_Company_research**: Emails seeking background information or due diligence on a company involved in a transaction (e.g., credibility, litigation history, public disputes, bankruptcies, financial health, track record).
    3.  **Intent_Comparison_LOI_Lease**: Emails asking to compare a Letter of Intent (LOI) with a final lease agreement to identify discrepancies, changes, or deviations in terms (e.g., TI allowances, common area maintenance, termination clauses).
    4.  **Intent_Lease_Abstraction**: Emails requesting the extraction of key lease metadata and clauses (e.g., rent, term, landlord, tenant, renewal options, escalation schedules, important dates, responsibilities).
    5.  **Intent_Lease_Listings_Comparison**: Emails asking to compare multiple lease listing summaries for properties, focusing on identifying the best terms, overlaps, gaps, per square foot pricing, and tenant-friendly clauses.
    6.  **Intent_Sales_Listings_Comparison**: Emails asking to compare multiple sales listing summaries for properties, focusing on metrics like pricing, square footage, capitalization rate (cap rate), and average price per square foot (PSF).
    7.  **Intent_Transaction_Date_navigator**: Emails focused on extracting, scheduling, or managing transaction-related dates (e.g., escrow, closing, notice periods, possession dates, due diligence deadlines, funding deadlines, inspection dates).

    **If an email clearly contains elements of more than one of the above intents, or if its primary intent does not fit any of the 8 categories, classify it as "Intent_Mixed_Other".** This "Intent_Mixed_Other" category is crucial for handling complex or out-of-scope requests.

    **Output Format:**

    For each email, provide *only* the most appropriate intent label. Do not include any additional text or explanation strictly.


    Here are some **Example Emails:**

    **Email 1:**
    Subject: Lease Summary for 123 Main St
    Body: Hi team, please summarize the key terms of the lease for the 123 Main St property. I need to know the base rent, commencement and expiry dates, renewal options, and escalation schedule. Thanks!

    **Classification 1:** Intent_Lease_Abstraction

    **Email 2:**
    Subject: LOI vs. Lease Discrepancies - 789 Oak Ave
    Body: Hey, I need help comparing the LOI we submitted for 789 Oak Ave with the final lease. Can you identify any deviations, especially around TI allowances and common area maintenance? Appreciate your help.

    **Classification 2:** Intent_Comparison_LOI_Lease

    **Email 3:**
    Subject: Review for Risky Clauses - New Lease for 456 Elm Rd
    Body: Could you please review the new lease for 456 Elm Rd and detect any potentially risky or missing lease clauses, such as those related to subletting rights or indemnity? Best regards.

    **Classification 3:** Intent_Clause_Protect

    **Email 4:**
    Subject: Background Check on Global Holdings Inc.
    Body: Urgent: Can you do a background check on Global Holdings Inc. before we proceed? I’m particularly interested in any litigation history or bankruptcies in the past 5 years. Cheers.

    **Classification 4:** Intent_Company_research


    Here is the email {email}. Now classify this keeping all points in view. Do not hallaucinate.
    """

    id2label = {
    0: "Intent_Amendment_Abstraction",
    1: "Intent_Clause_Protect",
    2: "Intent_Company_research",
    3: "Intent_Comparison_LOI_Lease",
    4: "Intent_Lease_Abstraction",
    5: "Intent_Lease_Listings_Comparison",
    6: "Intent_Sales_Listings_Comparison",
    7: "Intent_Transaction_Date_navigator"
  }

    encoded_input = tokenizer(email, return_tensors='pt', truncation=True, padding=True, max_length = 256)

    with torch.no_grad():
        output = model(**encoded_input)

    scores = output.logits[0].detach().numpy()
    print(scores)
    probs = softmax(scores)
    print(probs)


    predicted_class_id = probs.argmax()

    if predicted_class_id >=0 and predicted_class_id <=7 :
        predicted_label = id2label[predicted_class_id]
    else :
        predicted_label = "Intent_Mixed_Other"

    return f"\n🧠 Predicted Category: {predicted_label}"
