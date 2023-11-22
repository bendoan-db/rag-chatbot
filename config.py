# Databricks notebook source
# MAGIC %md 
# MAGIC ## Configuration file
# MAGIC
# MAGIC Please change your catalog and schema here to run the demo on a different catalog.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  --><img src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=config&demo_name=chatbot-rag-llm&event=VIEW">

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME="dbdemos_vs_endpoint"

DATABRICKS_SITEMAP_URL = "https://docs.databricks.com/en/doc-sitemap.xml"

catalog = "dbdemos"

email = spark.sql('select current_user() as user').collect()[0]['user']
username = email.split('@')[0].replace('.', '_')
dbName = db = f"rag_chatbot_{username}"
