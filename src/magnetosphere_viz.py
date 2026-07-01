def generate_magnetosphere_html(vsw, bz, log_flux):
    """Generate the magnetosphere visualization HTML, parameterized by real data values."""

    return f"""
    <div style="background:#05070d;border-radius:12px;padding:1rem;color:#e5e7eb;font-family:sans-serif">

    <div style="display:flex;gap:24px;margin-bottom:12px;flex-wrap:wrap">
      <div>
        <span style="font-size:12px;color:#9ca3af">Solar wind speed</span><br>
        <span style="font-size:15px;font-weight:600;color:#5dcaa5">{vsw:.0f} km/s</span>
      </div>
      <div>
        <span style="font-size:12px;color:#9ca3af">IMF Bz</span><br>
        <span style="font-size:15px;font-weight:600;color:#ed93b1">{bz:.1f} nT</span>
      </div>
      <div>
        <span style="font-size:12px;color:#9ca3af">log\u2081\u2080 flux</span><br>
        <span style="font-size:15px;font-weight:600;color:#facc77">{log_flux:.2f}</span>
      </div>
    </div>

    <div id="status-badge" style="display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;margin-bottom:10px;">NORMAL</div>

    <canvas id="scene" width="720" height="440" style="width:100%;border-radius:8px;display:block;background:#04060c"></canvas>

    </div>

    <script>
    (function(){{
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const earthX = W * 0.52, earthY = H * 0.5, earthR = 38;

    let vsw = {vsw};
    let bz = {bz};
    let flux = {log_flux};
    let particles = [];
    let t = 0;

    // Static starfield (generated once)
    const stars = [];
    for(let i=0;i<120;i++){{
      stars.push({{x:Math.random()*W, y:Math.random()*H, r:Math.random()*1.1+0.2, tw:Math.random()*6.28}});
    }}

    function spawnParticle(){{
      const speed = 1 + (vsw - 280) / 470 * 3.6;
      particles.push({{
        x: -10,
        y: earthY + (Math.random()-0.5) * 300,
        vx: speed,
        age: 0,
        captured: false,
        belt: Math.random() < 0.5 ? 1 : 2
      }});
    }}

    function fluxColor(f){{
      if (f >= 4.0) return {{r:226,g:75,b:74}};
      if (f >= 3.5) return {{r:239,g:159,b:39}};
      if (f >= 3.0) return {{r:250,g:199,b:117}};
      return {{r:93,g:202,b:165}};
    }}

    function fluxLabel(f){{
      if (f >= 4.0) return {{text:'SEVERE', bg:'rgba(226,75,74,0.15)', fg:'#E24B4A'}};
      if (f >= 3.5) return {{text:'HIGH', bg:'rgba(239,159,39,0.15)', fg:'#EF9F27'}};
      if (f >= 3.0) return {{text:'ELEVATED', bg:'rgba(250,199,117,0.15)', fg:'#FAC775'}};
      return {{text:'NORMAL', bg:'rgba(93,202,165,0.15)', fg:'#5dcaa5'}};
    }}

    function magnetopauseShape(angle, compression){{
      const baseR = 150 - compression * 40;
      const tailStretch = 1.0 + 0.35 * (1 - Math.cos(angle)) / 2;
      return baseR * (1 + 0.12*Math.cos(angle)) * tailStretch;
    }}

    function drawBelt(radius, fc, pulse, width){{
      ctx.strokeStyle = `rgba(${{fc.r}},${{fc.g}},${{fc.b}},${{0.5*pulse}})`;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.ellipse(earthX, earthY, radius, radius*0.4, 0, 0, Math.PI*2);
      ctx.stroke();
    }}

    function draw(){{
      ctx.fillStyle = '#04060c';
      ctx.fillRect(0,0,W,H);

      // starfield
      stars.forEach(s => {{
        const tw = 0.5 + 0.5*Math.sin(t*0.02 + s.tw);
        ctx.fillStyle = `rgba(200,215,240,${{0.25*tw}})`;
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2); ctx.fill();
      }});

      const compression = Math.max(0, -bz) / 20;
      const isReconnecting = bz < -3;

      // sunward direction indicator (solar wind arrow, left)
      ctx.strokeStyle = 'rgba(250,199,117,0.25)';
      ctx.lineWidth = 1;
      for(let i=0;i<3;i++){{
        const yy = earthY - 30 + i*30;
        ctx.beginPath(); ctx.moveTo(20, yy); ctx.lineTo(70, yy); ctx.stroke();
      }}
      ctx.fillStyle = 'rgba(250,199,117,0.4)';
      ctx.font = '10px sans-serif';
      ctx.fillText('solar wind', 20, earthY - 45);

      // magnetopause
      ctx.save();
      ctx.translate(earthX, earthY);
      ctx.beginPath();
      for(let a = -Math.PI; a <= Math.PI; a += 0.05){{
        const r = magnetopauseShape(a, compression);
        const x = Math.cos(a) * r * (a > Math.PI/2 || a < -Math.PI/2 ? 0.9 : 0.6);
        const y = Math.sin(a) * r * 0.45;
        if (a === -Math.PI) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }}
      ctx.closePath();
      ctx.strokeStyle = isReconnecting ? 'rgba(212,83,126,0.55)' : 'rgba(55,138,221,0.45)';
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.restore();

      const fc = fluxColor(flux);
      const beltPulse = 0.7 + 0.3 * Math.sin(t * 0.05 * (1 + flux/3));

      // outer glow halo
      const haloR = earthR + 60;
      const grad = ctx.createRadialGradient(earthX, earthY, earthR, earthX, earthY, haloR);
      grad.addColorStop(0, `rgba(${{fc.r}},${{fc.g}},${{fc.b}},0)`);
      grad.addColorStop(0.5, `rgba(${{fc.r}},${{fc.g}},${{fc.b}},${{0.30*beltPulse}})`);
      grad.addColorStop(1, `rgba(${{fc.r}},${{fc.g}},${{fc.b}},0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(earthX, earthY, haloR, haloR*0.42, 0, 0, Math.PI*2);
      ctx.fill();

      // TWO radiation belts (inner + outer Van Allen belts)
      drawBelt(earthR + 22, fc, beltPulse, 2.5);
      drawBelt(earthR + 46, fc, beltPulse*0.75, 2);

      // Earth with atmosphere rim
      ctx.strokeStyle = 'rgba(90,180,230,0.35)';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(earthX, earthY, earthR+3, 0, Math.PI*2); ctx.stroke();

      const earthGrad = ctx.createRadialGradient(earthX-10, earthY-10, 4, earthX, earthY, earthR);
      earthGrad.addColorStop(0, '#7fe0c0');
      earthGrad.addColorStop(0.55, '#1d9e75');
      earthGrad.addColorStop(1, '#063d31');
      ctx.fillStyle = earthGrad;
      ctx.beginPath();
      ctx.arc(earthX, earthY, earthR, 0, Math.PI*2);
      ctx.fill();

      // aurora caps during reconnection (storm visual)
      if (isReconnecting) {{
        const auroraA = 0.3 + 0.3*Math.sin(t*0.1);
        ctx.strokeStyle = `rgba(120,230,180,${{auroraA}})`;
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(earthX, earthY-earthR*0.55, earthR*0.5, Math.PI*0.9, Math.PI*2.1); ctx.stroke();
        ctx.beginPath(); ctx.arc(earthX, earthY+earthR*0.55, earthR*0.5, -Math.PI*0.1, Math.PI*1.1); ctx.stroke();
      }}

      if (Math.random() < 0.7 + vsw/1200) spawnParticle();

      particles.forEach(p => {{
        p.x += p.vx;
        p.age++;
        const dx = earthX - p.x;
        const dy = earthY - p.y;
        const dist = Math.sqrt(dx*dx+dy*dy);
        const mpR = magnetopauseShape(Math.atan2(dy,dx), compression);

        if (dist < mpR + 30 && !p.captured) {{
          if (isReconnecting && Math.random() < 0.03) {{
            p.captured = true;
          }}
        }}

        if (p.captured) {{
          const beltR = p.belt === 1 ? earthR+22 : earthR+46;
          const angle = t*0.03 + p.age*0.05;
          p.x = earthX + Math.cos(angle) * beltR;
          p.y = earthY + Math.sin(angle) * beltR * 0.4;
        }}

        const alpha = p.captured ? 0.85 : Math.max(0, 1 - p.age/300);
        ctx.fillStyle = p.captured ? `rgba(${{fc.r}},${{fc.g}},${{fc.b}},${{alpha}})` : `rgba(90,160,230,${{alpha*0.7}})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.captured ? 1.9 : 1.4, 0, Math.PI*2);
        ctx.fill();
      }});

      particles = particles.filter(p => p.x < W + 20 && p.age < 600);

      t++;
      requestAnimationFrame(draw);
    }}

    draw();

    const lbl = fluxLabel(flux);
    const badge = document.getElementById('status-badge');
    badge.textContent = lbl.text;
    badge.style.background = lbl.bg;
    badge.style.color = lbl.fg;
    }})();
    </script>
    """
