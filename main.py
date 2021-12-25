from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from simfinder import simQuotes


app = FastAPI()

template_dir = os.path.abspath(os.curdir)
template_dir = os.path.join(template_dir, 'templates')

templates = Jinja2Templates(directory=template_dir)

@app.get('/api')
async def home(request: Request):
    return templates.TemplateResponse('index.html',
     context={"request": request,"title": "Similar Quotes Finder" })


@app.post('/api/Similar_quotes')
async def similar_docs(request: Request, sim_: str = Form(...)):

    obj = simQuotes(sim_)
    result = obj.similarityFind()
    return templates.TemplateResponse('similar_docs.html',
                                     context={"request": request,"quote": sim_, "result": result})

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)