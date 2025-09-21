import streamlit as st

st.set_page_config(
    page_title="About - Songyue XIE",
)


st.sidebar.image("./assets/Hachiware.png", width=200)
st.sidebar.image("./assets/Chiikawa.png", width=200)
st.sidebar.image("./assets/Usage.png", width=200)



st.title("PROHI Dashboard by Songyue XIE")  # 请替换为你的真实姓名

st.markdown("---")


st.subheader("About DSHI Project")

st.markdown("""
I was exempted from the DSHI course
""")
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>PROHI Dashboard by Songyue XIE</p>
    </div>
    """,
    unsafe_allow_html=True
)