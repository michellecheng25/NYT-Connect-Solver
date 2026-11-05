import streamlit as st
from src.embeddings import WordVectors
from src.scorer import plain_similarity
from src.search import greedy_search
from src.data import load_puzzles
from src.evaluate import count_correct_groups


@st.cache_data
def get_puzzles():
    return load_puzzles()


st.title("Connections AI Solver")
st.write("A GloVe + greedy search baseline.")

puzzles = get_puzzles()

puzzle_index = st.selectbox(
    "Choose a puzzle",
    options=range(len(puzzles)),
    format_func=lambda i: (
        f"Puzzle {puzzles[i]['id']} — {puzzles[i]['date']}"
    ),
)

puzzle = puzzles[puzzle_index]

st.subheader(f"Puzzle — {puzzle['date']}")

for row in range(4):
    columns = st.columns(4)

    for column in range(4):
        index = row * 4 + column
        columns[column].info(puzzle["words"][index])




@st.cache_resource
def get_vectors():
    return WordVectors.load_glove()


if st.button("Run solver"):
    with st.spinner("Scoring groups and solving..."):
        wv = get_vectors()
        scores = plain_similarity(puzzle["words"], wv)
        predictions = greedy_search(scores)

    st.subheader("Predicted groups")

    for number, group in enumerate(predictions, start=1):
        words = [puzzle["words"][i] for i in sorted(group)]
        st.write(f"**Group {number}:** {', '.join(words)}")

    correct = count_correct_groups(predictions, puzzle["groups"])
    st.metric("Correct groups", f"{correct}/4")

    with st.expander("Show correct answers"):
        for name, group in zip(
            puzzle["group_names"], puzzle["groups"]
        ):
            words = [puzzle["words"][i] for i in sorted(group)]
            st.write(f"**{name}:** {', '.join(words)}")




st.divider()

if st.button("Evaluate first 10 puzzles"):
    with st.spinner("Evaluating 10 puzzles..."):
        wv = get_vectors()
        demo_puzzles = puzzles[:10]
        results = []

        for demo in demo_puzzles:
            scores = plain_similarity(demo["words"], wv)
            predictions = greedy_search(scores)
            correct = count_correct_groups(
                predictions, demo["groups"]
            )

            results.append({
                "Date": demo["date"],
                "Correct groups": correct,
            })

    total_correct = sum(row["Correct groups"] for row in results)
    fully_solved = sum(
        row["Correct groups"] == 4 for row in results
    )
    count = len(results)

    st.subheader(f"Results across {count} puzzles")
    st.caption("Preliminary results on the oldest puzzles.")
    st.dataframe(results, hide_index=True)

    st.metric(
        "Average correct groups",
        f"{total_correct / count:.2f}/4",
    )
    st.metric(
        "Fully solved",
        f"{fully_solved / count:.1%}",
    )