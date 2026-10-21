"use strict";
const labelInput=document.getElementById("label"), fileInput=document.getElementById("filename");
const plot=document.getElementById("plot"), plotStatus=document.getElementById("plot-status");
const audio=document.getElementById("sample-audio");
let recordings={};
function redraw() {
  const label=labelInput.value, filename=fileInput.value;
  plot.hidden=true;
  if(!filename){plotStatus.textContent="Записей в этом классе пока нет. Вернись к сбору датасета.";audio.removeAttribute("src");return;}
  plotStatus.textContent="Строю графики локально…";
  plotStatus.hidden=false;
  const url=`/api/plot?label=${encodeURIComponent(label)}&file=${encodeURIComponent(filename)}`;
  plot.onload=()=>{plot.hidden=false;plotStatus.hidden=true;};
  plot.onerror=()=>{plot.hidden=true;plotStatus.textContent="Не получилось построить графики. Проверь терминал.";};
  plot.src=url;
  audio.src=`/api/audio?label=${encodeURIComponent(label)}&file=${encodeURIComponent(filename)}`;
}
function choose() {
  const old=fileInput.value;
  fileInput.replaceChildren();
  for(const filename of (recordings[labelInput.value] || [])) {
    const opt=document.createElement("option");opt.value=filename;opt.textContent=filename;fileInput.add(opt);
  }
  if((recordings[labelInput.value] || []).includes(old))fileInput.value=old;
  redraw();
}
labelInput.addEventListener("change",choose);
fileInput.addEventListener("change",redraw);
fetch("/api/samples").then(r=>r.json()).then(data=>{recordings=data;choose();}).catch(e=>plotStatus.textContent=`Не удалось получить записи: ${e.message}`);
