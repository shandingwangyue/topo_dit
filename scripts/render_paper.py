"""Render the revised English paper with embedded fonts and typeset equations."""
from pathlib import Path
import re
import shutil
from html import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'pdf'/'topo_dit_revised.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
FONT_DIR=Path('C:/Windows/Fonts')
for name,file in [('Paper','arial.ttf'),('Paper-Bold','arialbd.ttf'),('Paper-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT_DIR/file)))
pdfmetrics.registerFontFamily('Paper',normal='Paper',bold='Paper-Bold',italic='Paper-Italic',boldItalic='Paper-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperBody',fontName='Paper',fontSize=10,leading=14.8,spaceAfter=8,textColor=colors.HexColor('#20252B')))
styles.add(ParagraphStyle(name='PaperTitle',fontName='Paper-Bold',fontSize=21,leading=26,spaceAfter=13,textColor=colors.HexColor('#17324D')))
styles.add(ParagraphStyle(name='PaperH1',fontName='Paper-Bold',fontSize=13,leading=17,spaceBefore=14,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='PaperH2',fontName='Paper-Bold',fontSize=11,leading=15,spaceBefore=9,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='Equation',fontName='Paper',fontSize=11,leading=20,alignment=TA_CENTER,spaceBefore=5,spaceAfter=12))
styles.add(ParagraphStyle(name='Reference',parent=styles['PaperBody'],fontSize=9,leading=13))

# Display equations are explicit typesetting equivalents of the LaTeX in the source.
EQUATIONS=[
 'P* = arg min<sub>P ≥ 0</sub> ∑<sub>i,k</sub> P<sub>ik</sub>C<sub>ik</sub> + ε ∑<sub>i,k</sub> P<sub>ik</sub>(log P<sub>ik</sub> − 1).',
 'P <b>1</b><sub>K</sub> = (1/N) <b>1</b><sub>N</sub>,　 P<sup>T</sup> <b>1</b><sub>N</sub> = (1/K) <b>1</b><sub>K</sub>.',
 'E<sub>k</sub> = (∑<sub>i</sub> A<sub>ik</sub>X<sub>i</sub>) / (∑<sub>i</sub> A<sub>ik</sub> + δ).',
 'v̂<sub>i</sub> = W<sub>o</sub>(∑<sub>k</sub> A<sub>ik</sub>F<sub>k</sub>) + b<sub>o</sub>.',
 'z<sub>τ</sub> = (1 − τ)z<sub>0</sub> + τz<sub>1</sub>,　 u = z<sub>1</sub> − z<sub>0</sub>.',
 'L<sub>FM</sub> = E[‖v<sub>θ</sub>(z<sub>τ</sub>, τ, c) − u‖<sub>2</sub><sup>2</sup>].',
 'z<sub>j+1</sub> = z<sub>j</sub> + (1/S)v<sub>θ</sub>(z<sub>j</sub>, j/S, c).',
 'O(ND<sup>2</sup> + NKD + INK + LK<sup>2</sup>D + LKD<sup>2</sup>).',
]

def inline(text):
    text=escape(text.replace('—','-').replace('–','-').replace('　',' '))
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'(https?://[^\s<]+)',r'<link href="\1" color="#245C8F">\1</link>',text)
    return text


def header_footer(c,doc):
    c.saveState()
    w,h=doc.pagesize
    c.setFont('Paper',8)
    c.setFillColor(colors.HexColor('#526273'))
    c.drawString(51,h-31,'TOPO-DiT  |  RESEARCH PROTOTYPE')
    c.drawRightString(w-51,h-31,'Revised 25 September 2026')
    c.setStrokeColor(colors.HexColor('#D5DEE6'))
    c.line(51,40,w-51,40)
    c.drawString(51,26,'Correctness verified on CPU; video quality not evaluated.')
    c.drawRightString(w-51,26,str(doc.page))
    c.restoreState()


def main():
    text=(ROOT/'topo_dit.md').read_text(encoding='utf-8')
    blocks=re.split(r'\n\s*\n',text.strip())
    story=[]; equation_index=0; references=False
    for block in blocks:
        block=block.strip()
        if block.startswith('$$'):
            if equation_index>=len(EQUATIONS):
                raise ValueError('new equation requires explicit PDF typesetting')
            story.append(Paragraph(EQUATIONS[equation_index].replace('　',' '),styles['Equation']))
            equation_index+=1
        elif block.startswith('# '):
            story.append(Paragraph(inline(block[2:]),styles['PaperTitle']))
        elif block.startswith('### '):
            story.append(Paragraph(inline(block[4:]),styles['PaperH2']))
        elif block.startswith('## '):
            references=block=='## References'
            story.append(Paragraph(inline(block[3:]),styles['PaperH1']))
        else:
            story.append(Paragraph(inline(block.replace('\n',' ')),styles['Reference' if references else 'PaperBody']))
    if equation_index!=len(EQUATIONS):
        raise ValueError('source equation count changed')
    doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=51,rightMargin=51,
                          topMargin=54,bottomMargin=55,title='Topo-DiT: Balanced Routing Bottlenecks for Latent Video Flow Matching',
                          author='Topo-DiT project',allowSplitting=True)
    doc.build(story,onFirstPage=header_footer,onLaterPages=header_footer)
    shutil.copy2(OUT,ROOT/'topo_dit.pdf')
    print(OUT)


if __name__=='__main__':
    main()
