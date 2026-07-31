Here is the complete Markdown and Python content for Lesson 2.2, continuing the "Smart Air Force Warfighter" theme. It is structured exactly like the previous lesson, ready to be dropped into your TeachBook repository.

---

# L28: Naive Bayes Classification

In the previous lesson, we used physical sensor data (numbers) to predict outcomes. But what happens when the intelligence we gather isn't numbers, but unstructured text?

Whether it's an intercepted enemy radio transmission, a secure network email, or a translated intelligence brief, the Air Force processes millions of words a day. To classify this text rapidly, we rely on **Bayes' Theorem**. The **Naive Bayes** classifier is incredibly fast, highly scalable, and forms the historical foundation for all modern natural language processing (NLP).

## The Core Concept: Bayes' Theorem

Bayes' Theorem allows us to update our beliefs based on new evidence. In classification, we want to find the probability of a specific class (like "Urgent") given the text evidence we just observed.

$$P(\text{Class} | \text{Evidence}) = \frac{P(\text{Evidence} | \text{Class}) \times P(\text{Class})}{P(\text{Evidence})}$$

* **$P(\text{Class})$ [The Prior]:** What is the baseline probability of this class before seeing any evidence? (e.g., 90% of all transmissions are Routine).
* **$P(\text{Evidence} | \text{Class})$ [The Likelihood]:** If the transmission *is* Urgent, how likely is it to contain the word "contact"?
* **$P(\text{Class} | \text{Evidence})$ [The Posterior]:** The final probability that the transmission is Urgent, given that we saw the word "contact".

---

## Text Classification

**Scenario:** Tactical Communications Triage. You are receiving thousands of short, intercepted radio transmissions. You need to automatically flag messages as "Routine" or "Urgent" so human analysts know what to read first.

### The Math: Naive Bayes for Text

To classify a full sentence, we look at the words $w_1, w_2, \dots, w_n$. The formula becomes:


$$P(\text{Class} | w_1, w_2, \dots, w_n) \propto P(\text{Class}) \prod_{i=1}^{n} P(w_i | \text{Class})$$

The $\propto$ symbol means "proportional to." We calculate this score for both "Urgent" and "Routine" and simply pick the class with the higher score.

> **Example 28.1 - Calculating the Posterior**
> Historically, 10% of transmissions are Urgent ($P(\text{U}) = 0.1$) and 90% are Routine ($P(\text{R}) = 0.9$).
> We intercept a new transmission: **"Enemy spotted"**.
> Let's look at our historical dictionary (Likelihoods):
> * $P(\text{``enemy''} | \text{U}) = 0.4$ (40% of Urgent messages use this word)
> * $P(\text{``spotted''} | \text{U}) = 0.3$
> * $P(\text{``enemy''} | \text{R}) = 0.01$ (1% of Routine messages use this word)
> * $P(\text{``spotted''} | \text{R}) = 0.05$
> 
> 
> **Calculate Urgent Score:** $0.1 \times 0.4 \times 0.3 = 0.012$
> **Calculate Routine Score:** $0.9 \times 0.01 \times 0.05 = 0.00045$
> **Result:** Since $0.012 > 0.00045$, the model confidently classifies the message as **Urgent**. Even though Routine messages are 9 times more common, the specific words provided overwhelming evidence to the contrary.

---

## Cyber Threats & The Independence Assumption

**Scenario:** Secure Network Phishing. You are filtering incoming emails on a secure Air Force network to detect malicious phishing attempts designed to steal credentials.

### The Math: The "Naive" Assumption

Why is it called *Naive* Bayes? Because the algorithm makes a massive, mathematically "naive" assumption: **It assumes every word in a sentence is completely independent of every other word.**

$$P(\text{"password"}, \text{"reset"} | \text{Malicious}) = P(\text{"password"} | \text{Malicious}) \times P(\text{"reset"} | \text{Malicious})$$

> **Example 28.2 - The Independence Failure**
> Let's look at the words "password" and "reset". In reality, if a phishing email contains the word "password", there is a massive probability the next word is "reset". They are highly correlated.
> However, Naive Bayes ignores this. It calculates the threat level of "password" (say, 80% malicious) and the threat level of "reset" (say, 70% malicious), and multiplies them together as if they were two entirely separate pieces of evidence.
>
> **Result:** This causes Naive Bayes to "double-count" the evidence, making its final probability scores overly confident (e.g., predicting a 99.99% probability of an attack). Yet, despite being mathematically overconfident, the *decision boundary* rarely changes—an email marked 99.99% malicious and an email marked 85% malicious are both correctly sent to the spam folder!

### EX: Email Phishing Detection

In this exercise, we will classify emails and visualize exactly *which* words are driving the model's decisions. You will execute the following steps:

* **Generate Data:** Create a dataset of Benign (Ham) and Malicious (Spam) emails.
* **Train Model:** Vectorize the text and train the Naive Bayes classifier.
* **Extract Log Probabilities:** Look inside the "brain" of the model to extract the mathematical weights (log probabilities) it assigned to every single word in the vocabulary.
* **Visualize:** Plot a horizontal bar chart showing the top 10 most "malicious" words to understand how the independence assumption manifests in the real world.

```python
import matplotlib.pyplot as plt

# 1. Synthetic Phishing Dataset
email_data = {
    'Email': [
        "meeting at 0900 regarding training", "URGENT: reset your network password immediately",
        "please review the attached briefing document", "click here to verify your secure login credentials",
        "lunch menu for the week is attached", "warning: unauthorized access detected click link to secure account",
        "reminder to submit your leave request", "your mailbox is full please verify account password"
    ],
    'Label': ['Benign', 'Malicious', 'Benign', 'Malicious', 'Benign', 'Malicious', 'Benign', 'Malicious']
}
df_emails = pd.DataFrame(email_data)

# 2. Train Model
vec_emails = CountVectorizer(stop_words='english') # Removes words like "the", "at", "your"
X_emails = vec_emails.fit_transform(df_emails['Email'])
nb_emails = MultinomialNB()
nb_emails.fit(X_emails, df_emails['Label'])

# 3. Extract Feature Log Probabilities
# nb_emails.feature_log_prob_[1] contains the probabilities for the 'Malicious' class
words = vec_emails.get_feature_names_out()
malicious_log_probs = nb_emails.feature_log_prob_[1] 

# Create a DataFrame for visualization
df_features = pd.DataFrame({'Word': words, 'Malicious_Log_Prob': malicious_log_probs})
df_top_malicious = df_features.sort_values(by='Malicious_Log_Prob', ascending=False).head(8)

# 4. Plot the most dangerous words
plt.figure(figsize=(8, 5))
plt.barh(df_top_malicious['Word'], df_top_malicious['Malicious_Log_Prob'], color='darkred')
plt.gca().invert_yaxis() # Put the highest probability at the top
plt.title('Top Words Driving "Malicious" Classification')
plt.xlabel('Log Probability (Closer to 0 is higher probability)')
plt.ylabel('Feature (Word)')
plt.show()

```

### Interpreting the Results

The bar chart reveals the inner workings of the model. Words like "password", "secure", and "verify" have the highest log probabilities (closest to 0) in the Malicious class.

Because of the naive independence assumption, if an adversary writes an email saying "verify password secure login", the model multiplies all these high probabilities together. It treats each word as a totally separate, damning piece of evidence. While a statistician would argue this math is technically flawed (because those words frequently appear together in the same sentence), the Cyber Warfare operator doesn't care—the model successfully quarantined the threat.

---

## Complex Problem: Zero-Frequency Events & Laplace Smoothing

**Scenario:** Adversary Communications. You are classifying intercepted text. What happens if the adversary uses a brand new code word (like "thunderbird") that has *never* appeared in your historical training data?

### The Math: The Zero Probability Problem

If a word has never been seen in a class, its historical probability is $0$. Because Naive Bayes uses multiplication, a single $0$ destroys the entire calculation:


$$P(\text{Urgent} | w_1, w_2, w_{new}) \propto 0.5 \times 0.3 \times \mathbf{0} = \mathbf{0}$$


Even if the rest of the message screams "Urgent", an unseen word acts as an absolute veto.

We fix this with **Laplace Smoothing** (also called Add-One smoothing). We mathematically pretend we have seen every possible word at least once.

$$P(w | C) = \frac{\text{count}(w, C) + \alpha}{\text{total words in } C + \alpha \times |V|}$$

* **$\alpha$ (Alpha):** The smoothing parameter (usually 1).
* **$|V|$:** The total size of our vocabulary.

> **Example 28.3 - Laplace Smoothing**
> Assume class "Urgent" has 100 total words. Our vocabulary $|V|$ contains 500 unique words.
> We encounter the unseen word "thunderbird".
> * **Without smoothing ($\alpha = 0$):** $P(\text{"thunderbird"} | U) = \frac{0}{100} = 0$
> * **With smoothing ($\alpha = 1$):** $P(\text{"thunderbird"} | U) = \frac{0 + 1}{100 + (1 \times 500)} = \frac{1}{600} \approx 0.0016$
> 
> 
> **Result:** The probability is tiny ($0.16\%$), representing that it is rare, but crucially, it is *not zero*. The math survives, and the rest of the sentence can still drive the classification.

### EX: Handling the Unknown

In this exercise, we will prove how smoothing saves the model from catastrophic failure when encountering new tactics. You will execute the following steps:

* **Create Unbalanced Data:** Train a model where the word "missile" appears in the Urgent class, but has *never* been seen in the Routine class.
* **Train Two Models:** Train one Naive Bayes model with smoothing (`alpha=1.0`) and one without (`alpha=1e-10` to approximate zero).
* **The Test:** Ask both models to classify a seemingly Routine message that inexplicably contains the word "missile".
* **Visualize:** Plot how the non-smoothed model assigns a literal 0% chance, while the smoothed model intelligently calculates a balanced probability.

```python
# 1. Specialized Dataset to force a Zero-Frequency event
data_smooth = {
    'Text': [
        "enemy missile launch detected", "urgent missile inbound", # 'missile' is in Urgent
        "routine patrol complete", "chow hall is open", "weather is clear" # 'missile' NEVER in Routine
    ],
    'Label': ['Urgent', 'Urgent', 'Routine', 'Routine', 'Routine']
}
df_smooth = pd.DataFrame(data_smooth)
vec_smooth = CountVectorizer()
X_smooth = vec_smooth.fit_transform(df_smooth['Text'])

# 2. Train Model WITHOUT Smoothing (alpha approaches 0)
nb_no_smooth = MultinomialNB(alpha=1e-10) 
nb_no_smooth.fit(X_smooth, df_smooth['Label'])

# 3. Train Model WITH Laplace Smoothing (alpha = 1)
nb_smoothed = MultinomialNB(alpha=1.0)
nb_smoothed.fit(X_smooth, df_smooth['Label'])

# 4. Test on a tricky sentence: A mostly routine sentence with the zero-frequency word.
tricky_sentence = ["routine patrol spotted a missile"] # 'missile' was never in routine training data
X_tricky = vec_smooth.transform(tricky_sentence)

prob_no_smooth = nb_no_smooth.predict_proba(X_tricky)[0]
prob_smoothed = nb_smoothed.predict_proba(X_tricky)[0]

# 5. Plot the comparison
labels = ['Probability Routine', 'Probability Urgent']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, prob_no_smooth, width, label='NO Smoothing (Alpha~0)', color='gray')
rects2 = ax.bar(x + width/2, prob_smoothed, width, label='WITH Smoothing (Alpha=1)', color='dodgerblue')

ax.set_ylabel('Calculated Probability')
ax.set_title('Impact of Laplace Smoothing on Unseen Words')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add values on top of bars
ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

plt.show()

```

### Interpreting the Results

Look at the gray bars (No Smoothing). When the model encounters the test sentence "routine patrol spotted a missile", it attempts to calculate the probability of the message being "Routine". However, because "missile" never appeared in a Routine message during training, the historical probability is 0. The multiplication chain breaks, and the model outputs an absolute **0.000** probability for Routine, forcing it to classify the message as Urgent, despite the presence of the word "routine".

Now look at the blue bars (With Smoothing). By adding a mathematically synthetic "1" to the counts (Laplace Smoothing), the probability of "missile" belonging to the Routine class is tiny, but non-zero. The math survives. The model can balance the weight of the word "routine" against the word "missile" and output a nuanced probability (e.g., 55% Routine, 45% Urgent), making it vastly more robust in dynamic, real-world warfare environments where adversaries constantly change their vocabulary.

---

## Knowledge Check

1. Why is the Naive Bayes algorithm considered "Naive"? Give an example of how this assumption might be violated in an intelligence report.
2. If an intercepted message contains the word "encryption", and this word has never appeared in your training dataset, what mathematical error will occur if you do not use smoothing?
3. Explain how Laplace (Add-One) smoothing prevents the mathematical failure identified in question 2.