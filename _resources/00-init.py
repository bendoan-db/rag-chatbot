# Databricks notebook source
# MAGIC %md 
# MAGIC # init notebook setting up the backend. 
# MAGIC
# MAGIC Do not edit the notebook, it contains import and helpers for the demo
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=00-init&demo_name=chatbot-rag-llm&event=VIEW">

# COMMAND ----------

# MAGIC %run ../config_DO_NOT_EDIT

# COMMAND ----------


dbutils.widgets.text("reset_all_data", "false", "Reset Data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf
import os
import langchain

if reset_all_data:
  print(f'clearing up db {dbName}')
  spark.sql(f"DROP DATABASE IF EXISTS `{dbName}` CASCADE")

# COMMAND ----------

def use_and_create_db(catalog, dbName, cloud_storage_path = None):
  print(f"USE CATALOG `{catalog}`")
  spark.sql(f"USE CATALOG `{catalog}`")
  spark.sql(f"""create database if not exists `{dbName}` """)

assert catalog not in ['hive_metastore', 'spark_catalog']
#If the catalog is defined, we force it to the given value and throw exception if not.
if len(catalog) > 0:
  current_catalog = spark.sql("select current_catalog()").collect()[0]['current_catalog()']
  if current_catalog != catalog:
    catalogs = [r['catalog'] for r in spark.sql("SHOW CATALOGS").collect()]
    if catalog not in catalogs:
      spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
      if catalog == 'dbdemos':
        spark.sql(f"ALTER CATALOG {catalog} OWNER TO `account users`")
  use_and_create_db(catalog, dbName)

if catalog == 'dbdemos':
  try:
    spark.sql(f"GRANT CREATE, USAGE on DATABASE {catalog}.{dbName} TO `account users`")
    spark.sql(f"ALTER SCHEMA {catalog}.{dbName} OWNER TO `account users`")
  except Exception as e:
    print("Couldn't grant access to the schema to all users:"+str(e))    

print(f"using catalog.database `{catalog}`.`{dbName}`")
spark.sql(f"""USE `{catalog}`.`{dbName}`""")    

# COMMAND ----------

# DBTITLE 1,Optional: Allowing Model Serving IPs
#If your workspace has ip access list, you need to allow your model serving endpoint to hit your AI gateway. Based on your region, IPs might change. Please reach out your Databrics Account team for more details.

# def allow_serverless_ip():
#   base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get(),
#   headers = {"Authorization": f"Bearer {<Your PAT Token>}", "Content-Type": "application/json"}
#   return requests.post(f"{base_url}/api/2.0/ip-access-lists", json={"label": "serverless-model-serving", "list_type": "ALLOW", "ip_addresses": ["<IP RANGE>"], "enabled": "true"}, headers = headers).json()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Helpers to get catalog and index status:

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# DBTITLE 1,endpoint
import time
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,index
def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue. Try deleting it? vsc.delete_index({index_full_name})')
            raise e
        return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 20 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pyspark.sql.types import StringType


def download_databricks_documentation_articles(max_documents=None):
    # Fetch the XML content from sitemap
    response = requests.get(DATABRICKS_SITEMAP_URL)
    root = ET.fromstring(response.content)

    # Find all 'loc' elements (URLs) in the XML
    urls = [loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    if max_documents:
        urls = urls[:max_documents]

    # Create DataFrame from URLs
    df_urls = spark.createDataFrame(urls, StringType()).toDF("url").repartition(10)

    # Pandas UDF to fetch HTML content for a batch of URLs
    @pandas_udf("string")
    def fetch_html_udf(urls: pd.Series) -> pd.Series:
        def fetch_html(url):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    return response.content
            except requests.RequestException:
                return None
            return None

        with ThreadPoolExecutor(max_workers=200) as executor:
            results = list(executor.map(fetch_html, urls))
        return pd.Series(results)

    # Pandas UDF to process HTML content and extract text
    @pandas_udf("string")
    def download_web_page_udf(html_contents: pd.Series) -> pd.Series:
        def extract_text(html_content):
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                article_div = soup.find("div", itemprop="articleBody")
                if article_div:
                    return str(article_div).strip()
            return None

        return html_contents.apply(extract_text)

    # Apply UDFs to DataFrame
    df_with_html = df_urls.withColumn("html_content", fetch_html_udf("url"))
    final_df = df_with_html.withColumn("text", download_web_page_udf("html_content"))

    # Select and filter non-null results
    final_df = final_df.select("url", "text").filter("text IS NOT NULL")
    if final_df.isEmpty():
      raise Exception("Dataframe is empty, couldn't download Databricks documentation, please check sitemap status.")

    return final_df

# COMMAND ----------

def display_gradio_app(space_name = "databricks-demos-chatbot"):
    displayHTML(f'''<div style="margin: auto; width: 1000px"><iframe src="https://{space_name}.hf.space" frameborder="0" width="1000" height="950" style="margin: auto"></iframe></div>''')

# COMMAND ----------

# DBTITLE 1,Temporary langchain wrapper - will be part of langchain soon
from typing import Any, Iterator, List, Optional

from langchain.pydantic_v1 import BaseModel
from langchain.schema.embeddings import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import AIMessage, BaseMessage
from langchain.adapters.openai import convert_message_to_dict


class DatabricksEmbeddings(Embeddings):
    def __init__(self, model: str, host: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.host = host

    def _query(self, texts: List[str]) -> List[List[float]]:
        os.environ['DATABRICKS_HOST'] = self.host
        def _chunk(texts: List[str], size: int) -> Iterator[List[str]]:
          for i in range(0, len(texts), size):
              yield texts[i : i + size]

        from databricks_genai_inference import Embedding

        embeddings = []
        for txt in _chunk(texts, 20):
            response = Embedding.create(model=self.model, input=txt)
            embeddings.extend(response.embeddings)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._query(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._query([text])[0]


class DatabricksChatModel(SimpleChatModel):
    model: str

    @property
    def _llm_type(self) -> str:
        return "databricks-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            from databricks_genai_inference import ChatCompletion
        except ImportError as e:
            raise ImportError("message") from e

        messages_dicts = [convert_message_to_dict(m) for m in messages]

        response = ChatCompletion.create(
            model=self.model,
            messages=[
                m
                for m in messages_dicts
                if m["role"] in {"system", "user", "assistant"} and m["content"]
            ],
            **kwargs,
        )
        return response.message
