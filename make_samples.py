from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path("samples")
OUT.mkdir(exist_ok=True)
styles = getSampleStyleSheet()

NOTE = ("SYNTHETIC TEST PAPER - hypothetical values created for testing ANVIKSHA. "
        "Not real experimental findings.")


def table(rows):
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    return t


def make(filename, title, blocks):
    story = [Paragraph(title, styles["Title"]), Paragraph(NOTE, styles["Italic"]), Spacer(1, 12)]
    for kind, content in blocks:
        if kind == "h":
            story.append(Paragraph(content, styles["Heading2"]))
        elif kind == "p":
            story.append(Paragraph(content, styles["BodyText"]))
        elif kind == "t":
            story += [Spacer(1, 6), table(content), Spacer(1, 10)]
    SimpleDocTemplate(str(OUT / filename), pagesize=A4).build(story)


BASE = [["Dataset", "Baseline", "SynthNet"]]
BASELINE = [87.0, 88.5, 87.2, 89.3, 88.0]


def acc_table(ours):
    rows = list(BASE)
    for i, (b, o) in enumerate(zip(BASELINE, ours), start=1):
        rows.append([f"D{i}", f"{b:.1f}", f"{o:.1f}"])
    return rows


# 1. clean paper: everything reproduces
make("paper_clean.pdf", "SynthNet: A Hypothetical Classifier (Test Paper 1)", [
    ("h", "Results"),
    ("p", "Table 1 reports accuracy (%) on five datasets."),
    ("t", acc_table([91.2, 92.8, 90.5, 93.1, 92.4])),
    ("p", "SynthNet achieves an average accuracy of 92.0% across the five datasets, "
          "improving over the baseline average of 88.0% by 4.0 points."),
    ("p", "On the test split, SynthNet correctly classified 462 of 500 images (92.4%). "
          "Precision is 0.90 and recall is 0.88, giving an F1 score of 0.89."),
])

# 2. discrepancy: stated average (92.4) does not match the table (mean = 91.7)
make("paper_discrepancy.pdf", "SynthNet-B: A Hypothetical Classifier (Test Paper 2)", [
    ("h", "Results"),
    ("p", "Table 1 reports accuracy (%) on five datasets."),
    ("t", acc_table([90.1, 92.3, 91.0, 92.8, 92.3])),
    ("p", "SynthNet-B achieves an average accuracy of 92.4% across the five datasets, "
          "outperforming the baseline average of 88.0% by 4.4 points."),
])

# 3. statistics: significance claim is not supported (Welch p is about 0.056)
make("paper_stats.pdf", "SynthNet-C: Stability Study (Test Paper 3)", [
    ("h", "Results"),
    ("p", "Over 10 independent runs, SynthNet-C attains 85.2 +/- 1.1 accuracy "
          "(mean +/- standard deviation, n = 10), versus 84.1 +/- 1.3 for the baseline "
          "(n = 10). The difference is statistically significant (p < 0.05, Welch's t-test)."),
    ("p", "The 95% confidence interval for the SynthNet-C mean accuracy is [84.4, 86.0]."),
])

# 4. not checkable: training / hardware claims only
make("paper_not_checkable.pdf", "DeepSynth: Large-Scale Training (Test Paper 4)", [
    ("h", "Results"),
    ("p", "A ResNet-50 trained for 200 epochs on CIFAR-100 reaches 94.1% top-1 accuracy."),
    ("p", "Inference is 3x faster than the baseline on a single A100 GPU."),
    ("p", "We believe the learned features are more interpretable than prior work."),
])

# 5. failure and replanning: data missing, repository does not exist
make("paper_failure.pdf", "SynthNet-E: Seeds and Supplementary Data (Test Paper 5)", [
    ("h", "Results"),
    ("p", "SynthNet-E reaches a mean accuracy of 90.3% over 5 random seeds. The per-seed "
          "results are listed in Table 3 of the supplementary material, which is not "
          "included in this document."),
    ("p", "Code: https://github.com/anviksha-demo/does-not-exist-repo"),
])
print("Created 5 sample papers in", OUT)
