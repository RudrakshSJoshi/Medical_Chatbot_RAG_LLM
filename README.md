# Medical Assistance Bot using RAG (Retrieval Augmented Generation) Architecture

This repository contains the implementation of a state-of-the-art **Medical Assistance Bot** leveraging **RAG (Retrieval Augmented Generation) Architecture**. The bot is designed to provide **accurate, relevant, and timely information on a wide range of medical topics, enhancing patient care and supporting healthcare professionals.** 

**Note:**
For configuring the repository locally and environment, refer to [CONFIG.md](CONFIG.md).
To see a test case, refer to [SAMPLE_OUTPUT.md](SAMPLE_OUTPUT.md).

## NEW UPDATES
- **Context Classifier Chatbot**
  - **Introduction**
    - The Context Classifier Chatbot is a sophisticated chatbot designed to streamline the process of understanding and categorizing user queries. This update introduces a context stripping and identifier mechanism, significantly enhancing the chatbot's efficiency and cost-effectiveness.

  - **Key Features**
    - **Context Stripping and Identifier**
      - **Pre-segregation Layer**: The chatbot utilizes a pre-segregation layer to categorize the meaning behind user questions into specific categories. Currently, it distinguishes between nutrition and health-related queries and general queries.
      - **Selective Context Retrieval**: The chatbot retrieves context only when the query is categorized as non-general (e.g., nutrition and health-related). This reduces the need for extensive context searches for general queries, optimizing performance.

    - **Benefits**
      - **Reduced Input Costs**: By categorizing queries and selectively retrieving context, the input costs are minimized. The input size remains almost constant for segregation purposes, and context retrieval is bypassed for general queries.
      - **Time Efficiency**: With a likelihood of general queries being between 20-80%, the chatbot saves time by avoiding unnecessary similarity searches and large context additions.
      - **Cost Savings**: The reduction in input prompt size and selective context retrieval contribute to overall cost savings.

    - **How It Works**
      - **User Query Segregation**: When a user submits a query, the chatbot first processes it through a pre-segregation layer.
      - **Category Identification**: The pre-segregation layer identifies whether the query is general or related to nutrition and health.
      - **Context Retrieval**: If the query is categorized as general, the chatbot responds without additional context retrieval. If it is related to nutrition and health, the necessary context is retrieved to provide a comprehensive response.

- **Vector Database Setup**
  There are now two different databases to integrate with chatbots, using
  1. ChromaDB
  2. FAISS (Facebook AI Similarity Search)

  - **Chroma Database Setup**
    - 384 dimensions (with `sentence-transformers/all-MiniLM-L6-v2`)
    - 1024 dimensions (with `Alibaba-NLP/gte-large-en-v1.5`)
    - 4096 dimensions (with `Cohere Embeddings`)

  - **FAISS Vector Database Setup**
    - 1024 dimensions (with `Alibaba-NLP/gte-large-en-v1.5` embedding functions)
    - FAISS setup includes:
      - `.faiss` and `metadata.json` files for medical and nutrition databases in two separate folders.

  - **Common Files**
    - A common `context_retrieve.py` file is present in the `vector_database_setup` folder, which provides instructions on retrieving data and contexts from the FAISS vector database, with open prompts to allow **Chain-Of-Thought** processing.

**Important Note:**
- FAISS vector database requires a pointer to files, the metadata provides the chunk location -> (name_of_file, page_no., chunk_id), and it should be fetched manually.

Key Features:
- Utilizes **RAG architecture** to combine retrieval of relevant documents with generative capabilities.
- Provides **precise and context-aware responses** based on extensive medical knowledge.
- Supports a variety of medical queries ranging from **symptoms and conditions to treatments and medications**.

## Knowledge Base

The Medical Assistance Bot is equipped with comprehensive knowledge based on the following books and topics: 

---

### 1. **Gerontological Nursing: Competencies for Care** by **Kristen L. Mauk**

- Foundations of Gerontological Nursing
- Communication with Older Adults
- Comprehensive Assessment and Skills
- Health Promotion and Disease Prevention in the Elderly
- Managing Illnesses and Health Conditions
- Assistive Technologies in Elder Care
- Ethical and Legal Issues in Gerontology
- Diversity and Intimacy in Aging
- Global Models of Health Care
- Interdisciplinary Team and Education
- Leadership and Future Trends

---

### 2. **Diagnostic and Statistical Manual of Mental Disorders** by **The American Psychiatric Association**

- Neurodevelopmental Disorders
- Schizophrenia Spectrum and Other Psychotic Disorders
- Bipolar and Related Disorders
- Depressive Disorders
- Anxiety Disorders
- Obsessive-Compulsive and Related Disorders
- Trauma- and Stressor-Related Disorders
- Dissociative Disorders
- Somatic Symptom and Related Disorders
- Feeding and Eating Disorders
- Elimination Disorders
- Sleep-Wake Disorders
- Sexual Dysfunctions
- Gender Dysphoria
- Disruptive, Impulse-Control, and Conduct Disorders
- Substance-Related and Addictive Disorders
- Neurocognitive Disorders
- Personality Disorders
- Paraphilic Disorders
- Other Mental Disorders
- Medication-Induced Movement Disorders and Other Adverse Effects of Medication
- Other Conditions That May Be a Focus of Clinical Attention
- Emerging Measures and Models

---

### 3. **Current Essentials of Medicine, Fourth Edition** by **Tierney, Saint and Whooley**

- Cardiovascular Diseases
- Pulmonary Diseases
- Gastrointestinal Diseases
- Hepatobiliary Disorders
- Hematologic Diseases
- Rheumatologic & Autoimmune Disorders
- Endocrine Disorders
- Infectious Diseases
- Oncologic Diseases
- Fluid, Acid–Base, and Electrolyte Disorders
- Genitourinary and Renal Disorders
- Neurologic Diseases
- Geriatrics
- Psychiatric Disorders
- Dermatologic Disorders
- Gynecologic, Obstetric, and Breast Disorders
- Common Surgical Disorders
- Common Pediatric Disorders
- Selected Genetic Disorders
- Common Disorders of the Eye
- Common Disorders of the Ear, Nose, and Throat
- Poisoning

---

### 4. **Disease Handbook for Childcare Providers** by **New Hampshire Department of Health and Human Services**

- Immunization Requirements
- Diseases That Are Preventable With Vaccines
- When A Child Should Be Excluded Or Dismissed
- When Staff Should Be Excluded
- What Diseases Must Be Reported To Health Officials
- Child Abuse
- Diapering Recommendations
- Pets In Daycare Facilities
- Food Handling For Childcare Settings
- Rashes

---

### 5. **Clinical Guidelines - Diagnosis and Treatment Manual** by **Dubois, Vasseur-Binachon, Yoshimoto**

- A Few Symptoms and Syndromes
- Respiratory Diseases
- Gastrointestinal Disorders
- Skin Diseases
- Eye Diseases
- Parasitic Diseases
- Bacterial Diseases
- Viral Diseases
- Genito-Urinary Diseases
- Medical and Minor Surgical Procedures
- Mental Disorders in Adults
- Other Conditions

---

### 6. **Indian First Aid Manual** by **Indian Red Cross Society**

- Basic First Aid Techniques
- Respiratory System and Breathing
- Heart, Blood Circulation, Shock
- Wounds and Injuries
- Bones, Joints and Muscles
- Nervous System and Unconsciousness
- Gastrointestinal Tract, Diarrhoea, Food Poisoning and Diabetes
- Skin, Burns, Heat Exhaustion, Fever and Hypothermia
- Poisoning
- Bites and Stings
- Senses, Foreign Bodies in Eye, Ear, Nose or Skin and Swallowed Foreign Objects
- Urinary System, Reproductive System and Emergency Childbirth
- Psychological First Aid
- Specific Emergency Situations and Disaster Management
- First Aid Techniques: Dressings, Bandages and Transport Techniques
- Content of a First Aid Kit

---

### 7. **Essentials of Human Nutrition** by **Mann and Truswell**

- Energy and Macronutrients
- Organic and Inorganic Essential Nutrients
- Nutrition-Related Disorders
- Foods
- Nutritional Assessment
- Life Stages
- Clinical and Public Health
- Case Studies

---

### 8. **Pediatric Nursing and Health Care** by **Ethiopia Public Health Training Initiative**

- Introduction to Child Health
- History Taking and Physical Examination
- Essential Nursing Care for Hospitalized Children
- Care of the New Born
- Congenital Abnormalities
- Normal Growth and Development
- Nutrition and Nutritional Deficiencies
- Acute Respiratory Infections
- Control of Diarrhea
- Systemic Diseases
- Vaccine Preventable Diseases
- Expanded Program on Immunization (EPI)
- Common Genetic Problems of Children

---