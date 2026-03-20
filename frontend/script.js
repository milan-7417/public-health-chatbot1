let historyList = document.getElementById("history")

let chatHistory = JSON.parse(localStorage.getItem("chatHistory")) || []

function saveHistory(){
localStorage.setItem("chatHistory", JSON.stringify(chatHistory))
}

function updateSidebar(){

historyList.innerHTML=""

chatHistory.forEach((chat,index)=>{

let li=document.createElement("li")
li.innerText=chat.question
li.onclick=()=>loadChat(index)

historyList.appendChild(li)

})
}

function loadChat(index){

let chatbox=document.getElementById("chatbox")
chatbox.innerHTML=""

let chat=chatHistory[index]

chat.messages.forEach(msg=>{

chatbox.innerHTML+=
`<div class="${msg.role}">
<span>${msg.text}</span>
</div>`

})
}

function clearHistory(){

chatHistory=[]
localStorage.removeItem("chatHistory")

updateSidebar()
document.getElementById("chatbox").innerHTML=""

}

function toggleSidebar(){
document.getElementById("sidebar").classList.toggle("hide")
}

/* ENTER KEY FOR MAIN CHAT */
function handleKey(event){
if(event.key==="Enter"){
sendMessage()
}
}

/* MAIN CHAT */
async function sendMessage(){

let message=document.getElementById("message").value
let language=document.getElementById("language").value

if(message.trim()==="") return

let chatbox=document.getElementById("chatbox")

chatbox.innerHTML+=
`<div class="user"><span>${message}</span></div>`

document.getElementById("message").value=""

try{

const response=await fetch("/chat",{

method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
query:message,
language:language
})

})

const data=await response.json()

chatbox.innerHTML+=
`<div class="bot"><span>${data.answer}</span></div>`

chatbox.scrollTop=chatbox.scrollHeight

let newChat={
question:message,
messages:[
{role:"user",text:message},
{role:"bot",text:data.answer}
]
}

chatHistory.push(newChat)
saveHistory()
updateSidebar()

}catch(error){

chatbox.innerHTML+=
`<div class="bot"><span>⚠️ Error connecting to server</span></div>`

}
}

/* ========================= */
/* REPORT SECTION */
/* ========================= */

async function uploadReport(){

let fileInput = document.getElementById("reportFile")
let file = fileInput.files[0]

if(!file){
alert("Please select a file")
return
}

let formData = new FormData()
formData.append("file", file)

const response = await fetch("/analyze-report",{
method:"POST",
body:formData
})

const data = await response.json()

document.getElementById("reportChatbox").innerHTML =
`<div class="bot"><span>${data.analysis}</span></div>`
}

async function sendReportMessage(){

let message = document.getElementById("reportMessage").value
let language = document.getElementById("language").value

if(message.trim()==="") return

let chatbox = document.getElementById("reportChatbox")

chatbox.innerHTML +=
`<div class="user"><span>${message}</span></div>`

document.getElementById("reportMessage").value=""

const response = await fetch("/report-chat",{

method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({
query:message,
language:language
})

})

const data = await response.json()

chatbox.innerHTML +=
`<div class="bot"><span>${data.answer}</span></div>`

chatbox.scrollTop = chatbox.scrollHeight
}

function handleReportKey(event){
if(event.key==="Enter"){
sendReportMessage()
}
}

async function deleteReport(){

await fetch("/delete-report",{
method:"POST"
})

document.getElementById("reportChatbox").innerHTML=""

alert("Report deleted successfully")
}

/* ========================= */
/* ALERTS */
/* ========================= */

async function loadAlerts(){

try{

const response=await fetch("/alerts")
const data=await response.json()

let alertBox=document.getElementById("alerts")
alertBox.innerHTML=""

data.alerts.forEach(alert=>{

let li=document.createElement("li")
li.innerHTML=`<a href="${alert.link}" target="_blank">${alert.title}</a>`

alertBox.appendChild(li)

})

}catch(error){

console.log("Alert loading failed")

}
}

/* INITIAL LOAD */

updateSidebar()
loadAlerts()
setInterval(loadAlerts,60000)