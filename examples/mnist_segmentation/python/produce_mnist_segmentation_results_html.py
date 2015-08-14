import os
import HTML
from ccm.Pages import Page

def image_link_code(img_URL):
    code = '<a><img src="%s"></a>' % img_URL
    return code

def html_page_code(page_code):
    code = '<!DOCTYPE html><html>%s</html>' % page_code
    return code
    
def html_head_code(title='title'):
    code = '<head><title>%s</title></head>\n' %  title
    return code

def html_body_code(body='HTML body'):
    code = '<body>%s</body>' % body
    return code

if __name__ == '__main__':
    
    example_dir = '/usr/local/google/home/zyan/proj/caffe/google_extra/examples/mnist_segmentation/'
    results_dir_URL = \
    'https://gfsviewer.corp.google.com/cns/ok-d/home/zyan/caffe/examples/mnist_segmentation/renet_1_lay/renet_1_lay_HP_nolbw_nopeep_seg_lr5e-2_results/'
    output_dir = example_dir + 'HTML/'
    output_html_name = 'mnist_segmentation_ReNet_1_layer_HP.html'
    
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    
    tb = HTML.Table(header_row=['Batch ID', 'Segmentation'])
    for i in range(200):
        tb.rows.append(['batch %d'%i, image_link_code(results_dir_URL + 'batch_%d.jpg' % i)])
    
    tb_htmlcode = str(tb)
    
    body_code = html_body_code(tb_htmlcode)
    page_code = html_page_code(html_head_code('mnist segmentaiton 1-layer ReNet HP') + body_code)
    with open(output_dir + output_html_name, 'w') as f:
        f.write(page_code)
    
    