document.addEventListener('DOMContentLoaded',()=>{
const form=document.getElementById('predict-form');
const result=document.getElementById('result');
const predText=document.getElementById('prediction-text');
const newBtn=document.getElementById('new');

// populate select menus by fetching /options
async function loadOptions(){
	try{
		const resp=await fetch('/options');
		const opts=await resp.json();
		const cropSel=document.getElementById('crop_type');
		const regionSel=document.getElementById('region');
		const soilSel=document.getElementById('soil_type');
		const addOptions=(sel,arr)=>{sel.innerHTML=''; if(!arr||arr.length===0){sel.innerHTML='<option value="">--select--</option>'; return;} arr.forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);});}
		addOptions(cropSel, opts.crop_type);
		addOptions(regionSel, opts.region);
		addOptions(soilSel, opts.soil_type);
	}catch(err){
		console.warn('Could not load options:', err);
	}
}

form.addEventListener('submit',async e=>{
	e.preventDefault();
	const data=new FormData(form);
	const payload={}; data.forEach((v,k)=>payload[k]=v);
	// ensure numeric fields are numbers
	payload.temperature_c = parseFloat(payload.temperature_c);
	payload.rainfall_mm = parseFloat(payload.rainfall_mm);
	payload.humidity_percent = parseFloat(payload.humidity_percent);

	predText.textContent='Predicting...'; result.hidden=false;
	try{
		const resp=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
		const json=await resp.json();
		if(resp.ok){
			predText.innerHTML=`<span class='pred'>${json.prediction_tonnes_per_hectare} tonnes/hectare</span>`;
		} else {
			predText.textContent='Error: '+(json.error||JSON.stringify(json));
		}
	}catch(err){ predText.textContent='Error: '+err.message; }
});

newBtn.addEventListener('click',()=>{ result.hidden=true; form.reset(); });

loadOptions();
});