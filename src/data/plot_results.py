import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_csv('./train_stats/ViT_Chess/ViT_Chess_DGX_V9.csv')
    epochs = df.shape[0]
    plt.style.use(["science", "grid"])
    print(df)

    # sns.set(style='darkgrid')
    plt.plot(df['Training Loss'], 'b-o', label="Training")
    plt.plot(df['Valid. Loss'], 'g-o', label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)

    plt.xticks(range(1, epochs + 1))
    plt.savefig("ViT_Loss.png")
    matplotlib.rcParams.update({"text.usetex": True})
    """
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )
    plt.savefig("ViT_Loss.pgf")
    """
