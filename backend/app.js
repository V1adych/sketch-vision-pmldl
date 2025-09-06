(function(){
  const btn = document.getElementById('btn');
  const countEl = document.getElementById('count');
  const cnv = document.getElementById('cnv');
  const ctx = cnv.getContext('2d');

  let count = 0;
  btn.addEventListener('click', () => {
    count += 1;
    countEl.textContent = String(count);
  });

  // Useless doodle: animated lines
  let t = 0;
  function draw(){
    const w = cnv.width, h = cnv.height;
    ctx.fillStyle = '#020617';
    ctx.fillRect(0,0,w,h);

    for(let i=0;i<60;i++){
      const x = (i * 5 + t) % w;
      const y = h/2 + Math.sin((i + t*0.05)) * 40;
      ctx.fillStyle = `hsl(${(i*6+t)%360} 80% 60%)`;
      ctx.fillRect(x, y, 3, 3);
    }

    t += 1;
    requestAnimationFrame(draw);
  }
  draw();
})();
