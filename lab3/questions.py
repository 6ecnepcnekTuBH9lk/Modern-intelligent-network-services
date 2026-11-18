def case(question, answer, page, quote, *alternatives):
    return {"question": question, "answer": answer,
            "evidence": [{"page": page, "quote": quote}] + [
                {"page": p, "quote": q} for p, q in alternatives]}


BANK = {
    "Q01": case(
        "What data were used for supervised fine-tuning in InstructGPT?",
        "Demonstrations of desired behavior written by human labelers.",
        1, "labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning"),
    "Q02": case(
        "Which InstructGPT model size was preferred to the 175B GPT-3 in human evaluations?",
        "The 1.3B parameter InstructGPT model.",
        1, "outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3",
        (2, "outputs from our 1.3B PPO-ptx model are preferred to those from the 175B GPT-3")),
    "Q03": case(
        "What three behavior goals are used to describe aligned language models?",
        "Helpfulness, honesty, and harmlessness.",
        2, "we want language models to be helpful (they should help the user solve their task), honest (they shouldn’t fabricate information or mislead the user), and harmless",
        (10, "models to be aligned if they are helpful, honest, and harmless")),
    "Q04": case(
        "What was used as the scalar reward in step 3 of the InstructGPT training procedure?",
        "The output of the reward model (RM).",
        6, "We use the output of the RM as a scalar reward.",
        (9, "it produces a reward determined by the reward model and ends the episode")),
    "Q05": case(
        "What was the inter-annotator agreement for held-out labelers, including uncertainty?",
        "77.3 ± 1.3%.",
        8, "for held-out labelers this number is 77.3 ± 1.3%."),
    "Q06": case(
        "What do β and γ control in the PPO-ptx objective?",
        "β controls the KL penalty; γ controls the pretraining gradients.",
        9, "The KL reward coefficient, β, and the pretraining loss coefficient, γ, control the strength of the KL penalty and pretraining gradients respectively."),
    "Q07": case(
        "How many labeler groups and folds were used to test reward-model generalization?",
        "Five groups and five-fold cross-validation.",
        12, "we split our labelers into 5 groups, and train 5 RMs (with 3 different seeds) using 5-fold cross validation"),
    "Q08": case(
        "On which two datasets did the appropriate pretraining mix coefficient reverse performance regressions?",
        "SQuADv2 and DROP.",
        15, "a value of the pretraining mix coefficient that both reverses the performance regressions on SQuADv2 and DROP"),
    "Q09": case(
        "What fraction of the InstructGPT prompt dataset was classified as English by langid.py?",
        "About 96%. The estimated true fraction may be 99% or higher.",
        35, "around 96% of our dataset (110k datapoints) is classified as English"),
    "Q10": case(
        "How many Reddit posts were used in the TL;DR summarization evaluation dataset?",
        "2,500 Reddit posts.",
        50, "2,500 Reddit posts to summarize."),
    "D01": case(
        "What is InstructGPT and how was it produced?",
        "A family of GPT-3 models fine-tuned using supervised demonstrations and RL from human feedback.",
        1, "fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT."),
    "D02": case(
        "What is the InstructGPT reward model trained to predict?",
        "Which output human labelers prefer for a given input.",
        6, "We then train a reward model to predict the human-preferred output.",
        (8, "we trained a model to take in a prompt and response, and output a scalar reward")),
    "D03": case(
        "What is PPO-ptx in InstructGPT training?",
        "PPO with pretraining gradients mixed into its updates.",
        9, "mixing the pretraining gradients into the PPO gradients, in order to fix the performance regressions on public NLP datasets. We call these models “PPO-ptx.”"),
    "D04": case(
        "What metric was used to evaluate TL;DR generated summaries?",
        "ROUGE-L against reference summaries.",
        50, "Summaries are judged via their ROUGE-L scores with respect to a set of reference summaries."),
}

MAIN = [f"Q{i:02d}" for i in range(1, 11)]
EARLY = ["Q01", "Q02", "Q03"]
LATE = ["Q05", "Q09", "Q10"]
BOUNDARY = ["Q04", "Q05", "Q06", "Q07", "Q08"]
CITED = ["Q02", "Q05", "Q06", "Q09", "Q10"]
PARAPHRASES = {
    "Q01": "What examples of desirable behavior did people supply for the first learning stage that created InstructGPT?",
    "Q05": "How often did the independent raters agree with one another, and what margin was reported?",
    "Q06": "In the method that continues to learn from the initial text corpus during policy improvement, what does each of the two weighting factors adjust?",
    "Q09": "What share of the inputs for this instruction-following project did the language detector identify as being written in English?",
    "Q10": "How many forum entries were there in the test of the model's ability to condense long messages?",
}

# Термины служат навигацией по полному PDF, а отсутствие ответа проверяется по смыслу.
ABSENT = {
    "N01": ("What SHA-256 checksum does the paper give for the trained 6B reward-model checkpoint?",
            ["sha-256", "sha256", "checksum", "checkpoint"], "Методы и приложение C описывают модель и выбор checkpoint, без контрольной суммы."),
    "N02": ("What exact CUDA driver patch version was used during InstructGPT training?",
            ["cuda", "driver version", "gpu"], "В тексте и приложении C есть параметры обучения, но версия драйвера не указана."),
    "N03": ("What was the p95 serving latency in milliseconds for the 1.3B InstructGPT on the API?",
            ["p95", "95th percentile", "latency", "milliseconds"], "Результаты API-оценки описывают предпочтения, а не задержку API."),
    "N04": ("What was the hourly carbon footprint in kg CO2e of PPO training?",
            ["carbon", "co2e", "emissions", "petaflops"], "Вычислительная стоимость обсуждается без почасовых выбросов CO2e."),
    "N05": ("What is the median waiting time in seconds between successive ranking tasks for an annotator?",
            ["waiting time", "median wait", "seconds", "labeler satisfaction"], "В приложении B есть процесс отбора и опрос разметчиков, но нет медианного интервала между заданиями."),
}

DIALOGUES = {
    "instruct": [("D01", BANK["D01"]["question"]), ("Q02", "Which size beat the 175B baseline?"),
                 ("Q01", "What examples were used for its initial supervised stage?")],
    "training": [("D03", BANK["D03"]["question"]), ("Q06", "What do its two coefficients control?"),
                 ("Q08", "On which two benchmarks did it reverse regressions?")],
    "summaries": [("D04", BANK["D04"]["question"]), ("Q10", "How many posts did that benchmark contain?"),
                  ("D04", "What was that score compared against?")],
}
