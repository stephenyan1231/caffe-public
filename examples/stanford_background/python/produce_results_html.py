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
    table_data = [
        ['Last name',   'First name',   'Age'],
        ['Smith',       'John',         30],
        ['Carpenter',   'Jack',         47],
        ['Johnson',     'Paul',         62],
    ]
    htmlcode = HTML.table(table_data)
    print htmlcode
    
    example_dir = '/google/src/cloud/zyan/caffe2/google3/third_party/caffe/google_extra/examples/stanford_background/'
    results_dir_URL = 'https://gfsviewer.corp.google.com/cns/ok-d/home/zyan/caffe/examples/stanford_background/renet_1_lay/renet_1_lay_no_lb_weight_lr-2_prediction/'
    output_dir = example_dir + 'HTML/'
    output_html_name = 'ReNet_1_layer.html'
    
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    img_list_file = example_dir + 'test_0.txt'
    with open(img_list_file) as f:
        img_names = []
        for line in f.readlines():
            img_names += [line[:-1]]
        print img_names
    
    
    tb = HTML.Table(header_row=['Image name', 'Segmentation'])
    for img_name in img_names:
        tb.rows.append([img_name, image_link_code(results_dir_URL + '%s.jpg' % img_name)])
    
    tb_htmlcode = str(tb)
    
    body_code = html_body_code(tb_htmlcode)
    page_code = html_page_code(html_head_code('1-layer ReNet Segmentation') + body_code)
    with open(output_dir + output_html_name, 'w') as f:
        f.write(page_code)
    
    