import tempfile
from typing import Optional, Type
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field, validator
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
    outdir: Optional[str] = "./datavizqa/static/"
    args_schema: Type[PythonPlotToolInput] = PythonPlotToolInput
    
    @validator("outdir")
    def outdir_validator(cls, v):
        if not v.endswith("/"):
            v = v + "/"
        return v

    def _run(self, query: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        _, output_image = tempfile.mkstemp(suffix=".png", dir=self.outdir)
        self.locals.update({"output_image": output_image})
        query = query.replace("plt.show()", "plt.savefig(output_image)")
        try:
            super()._run(query, run_manager)
        except Exception as err:
            print(err)
        output_image_path = "app/static/" + output_image.split("/")[-1]
        return output_image_path