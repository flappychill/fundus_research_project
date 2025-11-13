const form = document.getElementById("form")
const file = document.getElementById("file")
const btn = document.getElementById("btn")
const preview = document.getElementById("preview")
const result = document.getElementById("result")

form.addEventListener("submit", async (e)=>{
  e.preventDefault()
  if(!file.files[0]) return
  btn.disabled = true
  preview.innerHTML = ""
  result.innerHTML = ""
  const img = document.createElement("img")
  img.src = URL.createObjectURL(file.files[0])
  preview.appendChild(img)
  const fd = new FormData()
  fd.append("file", file.files[0])
  const res = await fetch("/predict",{method:"POST",body:fd})
  const data = await res.json()
  const head = document.createElement("h3")
  head.textContent = data.label + " (" + data.score.toFixed(3) + ")"
  result.appendChild(head)
  data.topk.forEach(x=>{
    const row = document.createElement("div")
    row.className = "prob"
    const lab = document.createElement("div")
    lab.textContent = x.label
    const bar = document.createElement("div")
    bar.className = "bar"
    const fill = document.createElement("div")
    fill.className = "fill"
    fill.style.width = Math.round(x.score*100) + "%"
    bar.appendChild(fill)
    row.appendChild(lab)
    row.appendChild(bar)
    result.appendChild(row)
  })
  btn.disabled = false
})
