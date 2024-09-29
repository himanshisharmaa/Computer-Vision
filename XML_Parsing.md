## What is XML?
- XML stands for Extensible Markup Language
- Similar to HTML in its appearance
- XML is used for data presentation
- It is exclusively designed to send and receive data

## What is parsing?
- Parsing means to read information from a file and split it into pieces by identifying parts of that particular XML file.

## Python XML parsing modules
1. **xml.etree.ElementTree**: 
    formats XML data in a tree structure which is the most natural representation of hierarchical data.

2. **xml.dom.minidom**:
    Used by people who are proficient with DOM(Document Object Module) DOM applications often stared by parsing XML And DOM.


###  xml.etree.ElementTree Properties
1. Tag: It is a string representing the type of data being stored
2. Attributes: Consists of a number of attributes stored as dictionaries.
3. Text String: A text string having information that needs to be displayed.
4. Tail String: Can also have tail strings if necessary.
5. Child Elements: Consists of a number of child elements stored as sequences.


### 1. Parsing the module
- parse()
    This function takes XML in file format to parse it.
    
    Example:

        import xml.etree.ElementTree as ET
        myTree= ET.parse("sample.xml")
        myroot=myTree.getroot()

- fromstring()
     Parses XML supplied as string parameter
     
     Example:

        import xml.etree.ElementTree as ET
        data='''
            <annotation>
                <folder>images</folder>
                <filename>image1.jpg</filename>
                <path>/path/to/images/image1.jpg</path>
                <source>
                    <database>Unknown</database>
                </source>
            </annotation>
            '''

            myroot=ET.fromstring(data)
            print(myroot.tag)

### 2. Finding Elements
We can find various elements and sub elements using tag,attrib,text,etc.

Example:

        import xml.etree.ElementTree as ET
        myTree= ET.parse("sample.xml")
        myroot=myTree.getroot()
        print(myroot.tag)
        print(myroot[0].tag)
        print("-----All tags-----")
        for x in myroot:
            print(x.tag)

Output:

        annotation
        folder
        -----All tags-----
        folder
        filename
        path
        source
        size
        segmented
        object


