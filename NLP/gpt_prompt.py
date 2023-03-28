'''
[`기존`]
새 질문 -> 독립형 질문 생성 -> 문서 내용 탐색 지시 -> 히스토리 기록

[standalone, 독립형 질문 생성 프롬프트]
{chat_history} = 히스토리, {question} = 새 질문
'Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question.
Provide answers in question language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:'


[문서 내용 탐색 지시 프롬프트]
{question} = 독립형 질문, {context} = 문서
'You are an AI assistant providing helpful advice.
You are given the following extracted parts of a long document and a question.
Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below.
Do NOT make up hyperlinks.
If you can't find the answer in the context below,
just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context,
politely respond that you are tuned to only answer questions that are related to the context.
Question: {question}
=========
{context}
=========
Answer in Markdown:'


[히스토리 생성]
history = ['유저 질문', 'GPT 대답']
'''

'''
[`수정`]
새 질문 -> 독립형 질문 생성 -> 문서 내용 탐색 지시 -> 히스토리 기록

[standalone, 독립형 질문 생성 프롬프트]
{chat_history} = 히스토리, {question} = 새 질문
'I give you a Chat History of someone and me chatting.
New Question is the next question following the chat history.
Please create a new "Standalone Question" that combines chat history and new questions.
Please reply in question language.
Chat History: {chat_history}
New Question: {question}'


[문서 내용 탐색 지시 프롬프트]
{question} = 독립형 질문, {context} = 문서
'I give you my question and the document.
You should only reply that reference the document below.
Generation of unreferenced answers is prohibited.
If you can't find the answer in the document below, just say "Hmm, I'm not sure."
Don't make up hyperlinks.
Please reply in question language.
Question: {question}
Document: {context}'


[히스토리 생성]
history = [f"me: {question}\n”, f”someone: {answer}\n”]


[통합형 프롬프트]
{chat_history} = 히스토리, {question} = 새 질문, {context} = 문서
'I give you chat history and my new question and the document.
You should only reply that reference the document below.
Generation of unreferenced answers is prohibited.
If you can't find the answer in the document below, just say "Hmm, I'm not sure."
Don't make up hyperlinks.
Please reply in question language.
Chat History: {chat_history}
My New Question: {question}
Document: {context}'
'''

'''
[`100단어 내외 요약 실험`]
요약 / 핵심 아이디어 / 키워드

질문: 
다음 3개의 문서를 요약하는 리포트를 100단어 내외로 작성해줘.
경제 연구원 말투로 부탁해.

1번 문서: 
2021년 임금근로자 대출 평균액이 1년 전보다 7% 증가한 5202만원으로 집계됐다. 20대 근로자 대출 증가세가 눈에 띄게 높아졌으며, 40대 근로자가 가장 많은 7638만원의 대출액을 가지고 있었다. 소득이 높을수록 대출액도 많아지는 추세이며, 대기업 근로자의 평균대출액은 중소기업 근로자의 1.9배를 기록했다. 다만, 금융당국의 가계부채 관리 강화로 대출 증가율은 둔화했다. 대출 연체율은 전 연령대에서 가장 높은 20대 임금근로자가 1.54%로 나타났다.
2번 문서: 
3월 중순 실리콘 밸리 은행의 파산은 유동성, 체계적, 실질적인 경기 순환 위기의 형태로 계속해서 전개되는 미국 지역 은행 위기를 촉발했습니다. 바이든 행정부는 도드-프랭크 법을 위기 대응의 근거로 삼아 위기가 시스템 위기로 번지는 것을 막기 위해 최선을 다하고 있습니다. 예금자 보호를 보장하고 책임져야 할 금융기관과 투자자에게 책임을 물으면서 유동성 위기를 해결하는 것이 최우선 과제입니다. 하지만, 신용 위기의 위험과 비정상적인 새로운 위험은 그 문제를 해결하기 어렵게 만듭니다. 게다가, 중국의 채권 매각은 미국 금융 시스템에 추가적인 압박을 가하고 있습니다.
3번 문서: 
반도체 산업에 대한 엇갈린 전망에도 불구하고 국내 투자자들은 반도체 테마의 상장지수펀드(ETF)에 대한 사랑을 잃을 기미를 보이지 않고 있습니다. 인공지능 개발은 반도체 시장의 장기적인 성장에 기여할 것으로 여겨지는데, 이 때문에 점점 더 많은 투자자들이 이런 유형의 투자로 눈을 돌리고 있습니다. 시가총액 상위 10개 반도체 기업에 투자하는 TIGER Fn 반도체 TOP10 ETF는 올해 개인투자자들이 128억원어치의 주식을 사들이며 순자산 1조원을 돌파했습니다. 다른 반도체 ETF도 상당한 유입을 경험해 대부분 올해 국내주식 ETF의 평균 수익률(11.64%)을 앞질렀습니다.

요약 리포트 내용: 
2021년 대출 증가율이 둔화되었지만, 소득이 높아질수록 대출액도 높아지는 추세를 보이며, 20대 대출 증가율이 눈에 띄게 높아졌다. 대출 연체율은 20대가 가장 높았다. 미국 지역 은행 위기가 촉발된 3월 중순 실리콘 밸리 은행 파산 사건은 유동성, 체계적, 실질적인 경기 순환 위기로 계속해서 전개되고 있으며, 바이든 행정부는 위기가 시스템 위기로 번지는 것을 막기 위해 최선을 다하고 있다. 반면에 국내 투자자들은 반도체 테마 ETF에 대한 관심을 계속 유지하고 있으며, TIGER Fn 반도체 TOP10 ETF는 개인 투자자들이 순자산 1조원을 돌파하는 등 상당한 인기를 끌고 있다.
'''