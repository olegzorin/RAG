from src import run_setfit2
from src.run_setfit2 import SetFitRequest, SetFitResponse

request = SetFitRequest(phrases=['Emergency situation', 'High UV index'])
response = SetFitResponse()

run_setfit2.SetFitExecutor().execute(request, response)

print(response.scores)