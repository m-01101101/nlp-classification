import matplotlib.pyplot as plt
import wordcloud

from colour import Color
from datarobot.models.word_cloud import WordCloud
from pandas.core.frame import DataFrame
from rich import box
from rich.table import Table
from typing import Optional


def df_to_table(
    pandas_df: DataFrame,
    rich_table: Table,
    show_index: bool = False,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    [
        rich_table.add_column(str(column), no_wrap=True)
        for column in pandas_df.columns
        if column not in rich_table.columns
    ]

    for index, value_list in enumerate(pandas_df.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    # Update the style of the table
    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD

    return rich_table


def word_cloud_plot(wc: WordCloud):
    # make color schema, red high, blue low
    colors = [Color("#2458EB")]
    colors.extend(list(Color("#2458EB").range_to(Color("#31E7FE"), 81))[1:])
    colors.extend(list(Color("#31E7FE").range_to(Color("#8da0a2"), 21))[1:])
    colors.extend(list(Color("#a18f8c").range_to(Color("#ffad9e"), 21))[1:])
    colors.extend(list(Color("#ffad9e").range_to(Color("#d80909"), 81))[1:])
    webcolors = [c.get_web() for c in colors]

    # filter out stopwords
    dict_freq = {
        wc_word["ngram"]: wc_word["frequency"]
        for wc_word in wc.ngrams
        if not wc_word["is_stopword"]
    }
    dict_coef = {
        wc_word["ngram"]: wc_word["coefficient"]
        for wc_word in wc.ngrams
        if not wc_word["is_stopword"]
    }

    def color_func(*args, **kwargs):
        word = args[0]
        palette_index = int(round(dict_coef[word] * 100)) + 100
        r, g, b = colors[palette_index].get_rgb()
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(
            int(r * 255), int(g * 255), int(b * 255)
        )

    wc_image = wordcloud.WordCloud(
        stopwords=set(),
        width=1400,
        height=1400,
        relative_scaling=0.5,
        prefer_horizontal=1,
        color_func=color_func,
        background_color=(0, 10, 29),
        font_path=None,
    ).fit_words(dict_freq)
    plt.imshow(wc_image, interpolation="bilinear")
    plt.axis("off")


def rebin_df(raw_df: DataFrame, number_of_bins: int) -> DataFrame:
    "rebin the output of the liftchart to n bins"
    cols = ["bin", "actual_mean", "predicted_mean", "bin_weight"]
    new_df = DataFrame(columns=cols)
    current_prediction_total = 0
    current_actual_total = 0
    current_row_total = 0
    x_index = 1
    bin_size = 60 / number_of_bins
    for rowId, data in raw_df.iterrows():
        current_prediction_total += data["predicted"] * data["bin_weight"]
        current_actual_total += data["actual"] * data["bin_weight"]
        current_row_total += data["bin_weight"]

        if (rowId + 1) % bin_size == 0:
            x_index += 1
            bin_properties = {
                "bin": ((round(rowId + 1) / 60) * number_of_bins),
                "actual_mean": current_actual_total / current_row_total,
                "predicted_mean": current_prediction_total / current_row_total,
                "bin_weight": current_row_total,
            }

            new_df = new_df.append(bin_properties, ignore_index=True)
            current_prediction_total = 0
            current_actual_total = 0
            current_row_total = 0
    return new_df


def matplotlib_lift(bins_df: DataFrame, bin_count: int, ax):

    dr_blue = "#1F77B4"
    dr_orange = "#FF7F0E"

    grouped = rebin_df(bins_df, bin_count)
    ax.plot(
        range(1, len(grouped) + 1),
        grouped["predicted_mean"],
        marker="+",
        lw=1,
        color=dr_blue,
        label="predicted",
    )
    ax.plot(
        range(1, len(grouped) + 1),
        grouped["actual_mean"],
        marker="*",
        lw=1,
        color=dr_orange,
        label="actual",
    )
    ax.set_xlim([0, len(grouped) + 1])
    ax.legend(loc="best")
    ax.set_title("Lift chart {} bins".format(bin_count))
    ax.set_xlabel("Sorted Prediction")
    ax.set_ylabel("Value")
    return grouped
