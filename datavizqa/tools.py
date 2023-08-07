import tempfile
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain.tools import PythonAstREPLTool


class PythonPlotToolInput(BaseModel):
    query: str = Field(description=(
        "A string formatted plain python script with imports and variables to execute."
    ))


class PythonPlotTool(PythonAstREPLTool):
    name = "python_plot"
    description = (
        "A data plotter. Use this to execute python commands for data visualization. "
        "When using this tool, sometimes output is abbreviated - "
        "Make sure it does not look abbreviated before using it in your answer. "
        "Don't add comments to your python code."
    )
    outdir: Optional[str] = "./datavizqa/static"
    args_schema: Type[PythonPlotToolInput] = PythonPlotToolInput

    def _run(self, query: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        _, output_image = tempfile.mkstemp(suffix=".png", dir=self.outdir)
        self.locals.update({"output_image": output_image})
        if "import matplotlib.pyplot as plt" not in query:
            query = "import matplotlib.pyplot as plt\n" + query
        if "plt.show()" not in query:
            query = query + "\nplt.show()"
        query = query.replace("plt.show()", "plt.savefig(output_image)")
        
        super()._run(query, run_manager)
        
        output_image_path = "app/static/" + output_image.split("/")[-1]
        return f"Chart saved at: {output_image_path}"