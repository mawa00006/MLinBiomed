import streamlit as st
import pandas as pd

st.title("DermaMNIST")
st.markdown("""The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of 
seven common pigmented skin lesions. Each image is of size 28x28x3. ([Source](https://medmnist.com)) \n""")

# Class 1
st.subheader("Class 0: Actinic keratoses and intraepithelial carcinoma")
col1, col2 = st.columns([1, 3])
with col1:
    # carousel(items=test_items)
    st.image('imgs/0.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Actinic keratosis is an abnormal growth of cells caused by long-term damage from the sun. The 
    small bumps typically appear on the parts of the body that are most exposed to the sun's rays such as the ears, 
    nose, cheeks, temples and bald scalp. They are not cancerous, but a small fraction of them will develop into skin 
    cancer (intraepithelial carcinoma).\" ([Source](https://www.yalemedicine.org/conditions/actinic-keratosis))""")

# Class 1
st.subheader("Class 1: Basal cell carcinoma")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/1.jpg', use_column_width=True)
with col2:
    st.markdown("""\"ABasal cell carcinoma is a type of skin cancer that occurs when there is damage to the DNA of basal
     cells in the top layer, or epidermis, of the skin. They are called basal cells because they are the deepest cells in
      the epidermis. In normal skin, the basal cells are less than one one-hundredth of an inch deep, but once a cancer
       has developed, it will spread deeper.\" ([Source](https://www.yalemedicine.org/conditions/basal-cell-carcinoma))""")

# Class 2
st.subheader("Class 2: Benign keratosis-like lesions")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/2.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Seborrheic keratosis is a type of benign (non-cancerous) skin tumor or growth. These 
    slow-growing spots are typically raised and sometimes have a rough texture. Some look similar to warts. They also 
    vary in number; some people have a single seborrheic keratosis, while others have several, dozens, 
    or even hundreds of spots on their skin. These growths are neither viral or bacterial and, therefore, 
    cannot spread.\" ([Source](https://www.yalemedicine.org/conditions/seborrheic-keratosis))""")

# Class 3
st.subheader("Class 3: Dermatofibroma")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/3.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Dermatofibroma is a benign skin condition characterized by the formation of firm, round nodules 
    in the dermis layer of the skin. These nodules are composed of fibrous tissue and are typically painless. 
    Dermatofibromas can occur anywhere on the body but are most commonly found on the legs.\" ([Source](
    https://www.yalemedicine.org/clinical-keywords/dermatofibroma))""")

# Class 4
st.subheader("Class 4: Melanoma")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/4.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Melanoma is a type of skin cancer that originates in the pigment-producing cells of the epidermis 
    called melanocytes. Of the three most common types of skin cancer, melanoma is the most dangerous..\" ([Source](
    https://www.yalemedicine.org/conditions/melanoma-treatment-options))""")

# Class 5
st.subheader("Class 5: Melanocytic nevi")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/5.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Melanocytic nevi are benign tumors that that arise in the skin. They have different sizes and 
    colors as outlined above. Benign nevi are usually round or oval-shaped and are uniform in color. There are more 
    nevi in areas of the body that have greater long-term exposure to the sun, such as the outer arm compared with 
    the inner arm.\" ([Source](https://www.yalemedicine.org/conditions/melanocytic-nevi-moles))""")

# Class 6
st.subheader("Class 6: Vascular lesions")
col1, col2 = st.columns([1, 3])
with col1:
    st.image('imgs/6.jpg', use_column_width=True)
with col2:
    st.markdown("""\"Vascular lesions are abnormal growths or malformations in the blood vessels, which can occur in 
    various parts of the body. They can be congenital or acquired and may result from injury, infection, 
    or other underlying medical conditions. Vascular lesions can range from harmless to potentially life-threatening, 
    depending on their location and severity.\" ([Source](
   https://www.yalemedicine.org/search?q=vascular+lesions))""")

st.header("Class Distributions")

st.markdown("""The DermaMNIST dataset consists of 10.015 samples which are split into Training/Validation/Testing set
  with a ratio of 7:1:2. This results in Training/Validation/Testing splits of size 7.007/1.003/2.005 respectively.""")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
val_df = pd.read_csv("val.csv")

# Store in dict to access easier later
df_dict = {
    "Train": train_df,
    "Test": test_df,
    "Val": val_df
}

# Select Split
split_select = st.multiselect("Select Splits to visualize",
                              ["Train", "Val", "Test"],
                              default=["Train", "Val", "Test"])

# At least one yplit has to be selected otherwise we cannot generate a plot
if len(split_select) == 0:
    st.markdown("**InputError**: Please select at least one of Train, Test, Val")
else:
    # Concatenate the selected splits
    selected_dfs = [df_dict[split] for split in split_select]
    plot_df = pd.concat(selected_dfs)

    # Get available classes
    available_classes = sorted(plot_df['Class'].unique().tolist())

    # Select Class
    class_select = st.multiselect("Select Classes to display",
                                  available_classes,
                                  default=available_classes)  # By default, all classes are selected

    # Filter the DataFrame based on the selected classes
    filtered_df = plot_df[plot_df['Class'].isin(class_select)]

    # Visualize data
    if not filtered_df.empty:
        st.bar_chart(filtered_df, x="Class", y="Count", color="Split", stack=False)
    else:
        st.markdown("No data to display for the selected classes.")

st.markdown("""#### Source Data: \n Philipp Tschandl, Cliff Rosendahl, et al., "The ham10000 dataset, 
a large collection of multisource dermatoscopic images of common pigmented skin lesions," Scientific data, vol. 5, 
pp. 180161, 2018.

Noel Codella, Veronica Rotemberg, et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by 
the International Skin Imaging Collaboration (ISIC)”, 2018, arXiv:1902.03368.""")
